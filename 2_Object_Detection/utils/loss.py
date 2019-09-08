#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from .match import *
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    ''' SSD の損失関数を計算するクラス '''
    
    def __init__(self, jaccard_threshold=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold  # match 関数の jaccard 係数の閾値
        self.negpos_ratio = neg_pos  # Hard Negative Mining の制限値
        self.device = device    # CPU or GPU で計算
        
        
    def forward(self, predictions, targets):
        """
        損失関数の計算
        
        Parameters
        ----------
        predictions : (loc, conf, dbox_list)
            SSD の訓練時の出力
            loc : torch.Size([num_batch , 8732, 4])
            conf : torch.Size([num_batch , 8732, 21])
            dbox_list : torch.Size([8732, 4])
        targets : [num_batch , num_objs, 5]
            5 は正解のアノテーション情報 [xmin, ymin, xmax, ymax, label_ind]
            
        Returns
        -------
        loss_l : tensor
            loc の損失値
        loss_c : tensor
            conf の損失値
        """
        
        # SSD モデルの出力はタプルなので分解
        loc_data, conf_data, dbox_list = predictions
        
        # 要素数のカウント
        num_batch = loc_data.size(0)    # ミニバッチサイズ
        num_dbox = loc_data.size(1)     # デフォルトボックスの数 = 8732
        num_classes = conf_data.size(2) # クラス数 = 21
        
        # 損失の計算に使用するものを格納する変数を作成
        # 各デフォルトボックスに一番近い正解のバウンディングボックスのラベルを格納させる
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        # 各デフォルトボックスに一番近い正解のバウンディングボックスの位置情報を格納させる
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)
        
        # loc_t と conf_t_label にデフォルトボックスと targets を match させた結果を上書き
        for idx in range(num_batch):
            # 現在のミニバッチの正解アノテーションのバウンディングボックスとラベルを取得
            truths = targets[idx][:, :-1].to(self.device)
            labels = targets[idx][:, -1].to(self.device)
            
            dbox = dbox_list.to(self.device)
            # 関数 match を実行し、loc_t と conf_t_label の内容を更新する
            # loc_t: 各デフォルトボックスに一番近い正解のバウンディングボックスの位置情報が上書きされる
            # conf_t_label:各デフォルトボックスに一番近いバウンディングボックスのラベルが上書きされる
            # ただし、一番近いバウンディングボックスとの jaccard overlap が 0.5 より小さい場合は正解バウンディングボックスのラベル conf_t_label は背景クラスの 0 とする
            variance = [0.1, 0.2]  # バウンディングボックスに変換するときの係数
            match(self.jaccard_threshold, truths, dbox, variance, labels, loc_t, conf_t_label, idx)
            
        #---------------------------------------------------------------------------------------------------
        # 位置の損失:loss_l を計算
        # Smooth L1 関数で損失を計算する。ただし、物体を発見したデフォルトボックスのオフセットのみを計算する
        #---------------------------------------------------------------------------------------------------
        
        # 物体を検出したバウンディングボックスを取り出すマスクを作成
        pos_mask = conf_t_label > 0
        # pos_mask を loc_data のサイズに変形
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        # Positive なデフォルトボックスと教師データ loc_t を取得
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        # Positive なデフォルトボックスの損失（誤差）を計算
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        
        #------------------------------------------------------------------------------------------------------------------
        # クラス予測の損失 loss_c の計算
        # 交差エントロピー誤差関数で損失を計算する。ただし、背景クラスが正解であるデフォルトボックスが圧倒的に多いので、
        # Hard Negative Mining を実施し、物体発見デフォルトボックスと背景クラスデフォルトボックスの比が 1:3 になるようにする
        # そこで背景クラスデフォルトボックスと予想したもののうち、損失が小さいものは、クラス予測の損失から除く
        # ------------------------------------------------------------------------------------------------------------------
        batch_conf = conf_data.view(-1, num_classes)
        
        # クラス予測の損失を関数を計算（reduction='none' にして、和をとらず次元をつぶさない）
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")
        
        # Hard Negative Mining で抽出するデフォルトボックスのマスクを作成
        # Positive なデフォルトボックスの損失を0にする
        num_pos = pos_mask.long().sum(1, keepdim=True) # ミニバッチごとの物体クラス予測の数
        loss_c = loss_c.view(num_batch, -1)              # torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0                           # 物体のあるデフォルトボックスは損失を0にする
        
        # Hard Negative Mining を実行
        # 各デフォルトボックスの損失の大きさ loss_c の順位 idx_rank を求める
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        
        # Hard Negative Mining で残すデフォルトボックスの数を決める
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)
        
        # num_neg よりも損失の大きいデフォルトボックスを抽出するマスクを作る
        # torch.Size([num_batch, 8732])
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)
        
        # マスクの形を conf_data に揃える（torch.Size([num_batch, 8732]) => torch.Size([num_batch, 8732, 21])）
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)  # Positive なデフォルトボックスの conf を取り出すマスク
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)  # Negative なデフォルトボックスの conf を取り出すマスク
        
        # conf_data から pos と neg だけを取り出して conf_hnm とする（torch.Size([num_pos + num_neg, 21）
        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)
        
        # conf_t_label から pos と neg だけを取り出して conf_t_label_hnm とする
        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]
        
        # confidence の損失関数を計算
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction="sum")
        
        # 物体を発見したバウンディングボックスの数（全ミニバッチの合計）で損失を割り算
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        
        return loss_l, loss_c
