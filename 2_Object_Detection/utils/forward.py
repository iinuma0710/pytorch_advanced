#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from math import sqrt
from itertools import product
from .network import *


def decode(loc, dbox_list):
    """
    オフセット情報を用いてデフォルトボックスからバウンディングボックスに変換する
    
    Parameters
    ----------
    loc: [8372, 4]
        SSD モデルで推論するオフセット情報
    dbox_list: [8732, 4]
        デフォルトボックスの情報
        
    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        バウンディングボックスの情報
    """
    
    # オフセット情報からバウンディングボックスを求める
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)
    ), dim=1)
    
    # [xmin, ymin, xmax, ymax] の形に変形する
    boxes[:, :2] -= boxes[:, 2:] / 2  # (xmin, ymin) の計算
    boxes[:, 2:] += boxes[:, :2]      # (xmax, ymax) の計算
    
    return boxes


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Surpression を行う関数
    boxes のうち一定以上の overlap しているバウンディングボックスを削除
    
    Parameters
    ----------
    boxes : [確信度閾値（0.01）を超えたバウンディングボックス数, 4]
        バウンディングボックスの情報
    scores : [確信度閾値（0.01）を超えたバウンディングボックス数]
        conf の情報
        
    Returns
    -------
    keep : リスト
        conf の降順に NMS を通過した index が格納される
    count : int
        NMS を通過したバウンディングボックスの数
    """
    
    # torch.Size([ 確信度閾値を超えたバウンディングボックス数 ])、要素は全部 0
    keep  = scores.new(scores.size(0)).zero_().long()
    count = 0
    
    # 各バウンディングボックスの面積を計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    
    # boxes をコピーしてバウンディングボックスの IOU の計算に使う
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()
    
    # scores を昇順にソート
    v, idx = scores.sort(0)
    
    # 上位 top_k 個（200個）のバウンディングボックスのインデックスを取り出す
    # ただし，200個ない場合もある
    idx = idx[-top_k:]
    
    # idx の要素数が0でない限りループ
    while idx.numel() > 0:
        i = idx[-1]  # 最大の conf のインデックスを取得
        
        # keep の末尾に i を格納
        # このインデックスのバウンディングボックスと大きく被っているバウンディングボックスをここから削除
        keep[count] = i
        count += 1
        
        # idx の要素を人るずつ減らし，残り1個になったらループを抜ける
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        
        # ----------------------------------------------------------------------------------------------
        # これから keep に格納したバウンディングボックスと被りの大きいバウンディングボックスを抽出し除去
        # ----------------------------------------------------------------------------------------------
        
        # i までのバウンディングボックスを out に指定した変数として保存
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)
        
        # 全てのバウンディングボックスについて，index が i のバウンディングボックスの範囲に限定
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])
        
        # w と h のテンソルサイズを index を1つ減らした大きさにする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)
        
        # clamp した状態でバウンディングボックスの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        
        # clamp された状態での面積を求める
        inter = tmp_w * tmp_h
        
        # IOU を計算
        rem_areas = torch.index_select(area, 0, idx)  # 各バウンディングボックスの元の面積
        union = (rem_areas - inter) + area[i]         # 2つのエリアの AND の面積
        IoU = inter / union
        
        # IOU が overlap より小さい idx のみを残す
        idx = idx[IoU.le(overlap)] # le は Less than or Equal to の処理をする演算
        
    return keep, count


class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax  = nn.Softmax(dim=-1) # conf をソフトマックス関数で正規化する
        self.conf_thresh = conf_thresh     # conf_thresh より大きいデフォルトボックスのみ扱う
        self.top_k = top_k                 # conf の top_k 個だけを計算に使用する
        self.nms_thresh = nms_thresh       # nms_thresh より大きいものは同一物体へのバウンディングボックスとみなす
        
    def forward(self, loc_data, conf_data, dbox_list):
        """
        順伝搬の計算を実行する。
        
        Parameters
        ----------
        loc_data: [batch_num,8732,4]
            オフセット情報。
        conf_data: [batch_num, 8732,num_classes]
            検出の確信度。
        dbox_list: [8732,4]
           デフォルトボックス の情報
        
        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            (batch_num、クラス、conf の top200、BBox の情報)
        """

        # 各サイズを取得
        num_batch = loc_data.size(0) # ミニバッチのサイズ
        num_dbox = loc_data.size(1)  # デフォルトボックスの数
        num_classes = conf_data.size(2)   # クラス数
        
        # conf はソフトマックスを適用して正規化
        conf_data = self.softmax(conf_data)
        
        # 出力の型を作成
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        # conf_data を [batch_num,8732,num_classes] から [batch_num, num_classes,8732] にする
        conf_preds = conf_data.transpose(2, 1)
        
        # ミニバッチごとのループ
        for i in range(num_batch):
            # 1. loc と DBox から修正した BBox [xmin, ymin, xmax, ymax] を求める
            decoded_boxes = decode(loc_data[i], dbox_list)
            
            # conf のコピーを作成
            conf_scores = conf_preds[i].clone()
            
            # 画像クラスごとのループ(背景クラスの index である 0 は計算せず、index=1 から)
            for cl in range(1, num_classes):
                # 2.conf の閾値を超えた BBox を取り出す
                # conf の閾値を超えているかのマスクを作成し、閾値を超えた conf のインデックスを c_mask として取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                
                # gt は Greater than のこと。gt により閾値を超えたものが 1 に、以下が 0 になる
                # conf_scores: torch.Size([21, 8732])
                # c_mask: torch.Size([8732])
                
                # scores は torch.Size([ 閾値を超えた BBox 数 ])
                scores = conf_scores[cl][c_mask]
                
                # 閾値を超えた conf がない場合、つまり scores=[] のときは、何もしない
                if scores.nelement() == 0:
                    continue
                    
                # c_mask を decoded_boxes に適用できるようにサイズを変更します
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                
                # l_mask を decoded_boxes に適応します
                # decoded_boxes[l_mask] で 1 次元になってしまうので、view で(閾値を超えた BBox 数 , 4)サイズに変形しなおす
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # 3. Non-Maximum Suppression を実施し、被っている BBox を取り除く
                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                
                # output に Non-Maximum Suppression を抜けた結果を格納
                output[i, cl, :count] = torch.cat(( scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        
        # SSD のネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])
        
        # デフォルトボックスの作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()
        
        # 推論時は Detect クラスを用意する
        if phase == 'inference':
            self.detect = Detect()
            
    def forward(self, x):
        sources = list() # source1 〜 6 を格納
        loc = list()     # loc の出力を格納
        conf = list()    # conf の出力を格納
        
        # vgg の conv4_3 までを計算
        # conv4_3 の出力を L2Norm に入力し、source1 を作成、sources に追加
        for k in range(23):
            x = self.vgg[k](x)
        source1 = self.L2Norm(x)
        sources.append(source1)
        
        # vgg を最後まで計算し、source2 を作成、sources に追加
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        
        # extras の conv と ReLU を計算し，source3 〜 6 を sources に追加
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
                
        # source1 〜 6 に、それぞれ対応する畳み込みを 1 回ずつ適用する
        # zip で for ループの複数のリストの要素を取得
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permute は要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            
            # l(x) と c(x) で畳み込みを実行
            # l(x) と c(x) の出力サイズは [batch_num, 4 * アスペクト比の種類数 featuremap の高さ , featuremap 幅 ]
            # source によって、アスペクト比の種類数が異なり、面倒なので順番入れ替えて整える
            # permute で要素の順番を入れ替え [minibatch 数 , featuremap 数 , featuremap 数 ,4 * アスペクト比の種類数 ] へ
            # (注釈)
            # torch.contiguous() はメモリ上で要素を連続的に配置し直す命令で，あとで view 関数を使用する
            # この view を行うためには、対象の変数がメモリ上で連続配置されている必要がある
            
        # さらに loc と conf の形を変形
        # loc のサイズは、torch.Size([batch_num, 34928])
        # conf のサイズは torch.Size([batch_num, 183372]) になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        # さらに loc と conf の形を整える
        # loc のサイズは、torch.Size([batch_num, 8732, 4])
        # conf のサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # 最後に出力する
        output = (loc, conf, self.dbox_list)

        if self.phase == "inference": # 推論時
            # クラス「Detect」の forward を実行
            # 返り値のサイズは torch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])
        else: # 学習時
            # 返り値は (loc, conf, dbox_list) のタプル
            return output

