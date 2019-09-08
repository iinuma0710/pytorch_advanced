#!/usr/bin/env python
# coding: utf-8


import torch
import pandas as pd
from torch import nn
from numpy import sqrt
from itertools import product


def make_vgg():
    ''' 34層の VGG モジュールを作成する '''
    layers = []
    in_channels = 3  # 入力の色チャネル
    
    # VGG モジュールで使う畳み込み層やマックスプーリングのチャネル数
    # M  : マックスプーリング（出力テンソルのサイズは floor モード）
    # MC : マックスプーリング（出力テンソルのサイズは ceil モード）
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'MC', 512, 512, 512, 'M', 512, 512, 512]
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    
    return nn.ModuleList(layers)


# 8層の extras モジュールを作成
def make_extras():
    layers =[]
    in_channels = 1024
    
    # extras モジュールの畳み込み層のチャネル数を設定するコンフィギュレーション
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]
    
    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]
    
    return nn.ModuleList(layers)


def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []
    
    # VGG の22層目，conv4_3（source1）に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]
    # VGG の最終層（source2）に対する畳み込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]
    # extras（source3）に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]
    # extras（source4）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]
    # extras（source5）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]
    # extras（source6）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]
    
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


class L2Norm(nn.Module):
    def __init__(self, input_cannels=512, scale=20):
        super(L2Norm, self).__init__()  # 親クラスのコンストラクタを実行
        self.weight = nn.Parameter(torch.Tensor(input_cannels))
        self.scale = scale  # 係数 weight の初期値として設定する値
        self.reset_parameters() # パラメータの初期化
        self.eps = 1e-10
        
    def reset_parameters(self):
        ''' 結合パラメータを大きさ scale の値にする初期化を実行 '''
        nn.init.constant_(self.weight, self.scale)  # weight の値が全て scale になる
        
    def forward(self, x):
        """
        38x38 の特徴量に対して512チャネルについて二乗和の平方根を求めた 38x38 個の値を使用し，
        各特徴量を正規化してから係数を掛け算する
        """
        
        # 各チャネルについて正規化，テンソルサイズは torch.Size([batch_num, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        
        # 係数（torch.Size([512])）を掛ける
        # torch.Size([batch_num, 512, 38, 38]) のテンソルに変形
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x
        
        return out


# デフォルトボックスを出力するクラスの実装
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        
        # 初期設定
        self.image_size = cfg['input_size']        # 画像サイズ 300
        self.feature_maps = cfg['feature_maps']    # [38, 19, ...] 各 source も特徴量マップのサイズ
        self.num_priors = len(cfg["feature_maps"]) # source の個数 6
        self.steps = cfg['steps']                  # [8, 16, ...] デフォルトボックスのピクセルサイズ
        self.min_sizes = cfg['min_sizes']          # [30, 60, ...] 小さい正方形のデフォルトボックスのピクセルサイズ
        self.max_sizes = cfg['max_sizes']          # [60, 111, ...] 大きい正方形のデフォルトボックスのピクセルサイズ
        self.aspect_ratios = cfg['aspect_ratios']  # 長方形のデフォルトボックスのアスペクト比
        
    def make_dbox_list(self):
        ''' デフォルトボックスの作成 '''
        mean = []
        # 'feature_maps' : [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # f までの数で2ペアの組み合わせを作る f_P_2
                
                # 特徴量の画像サイズ
                # 300 / 'steps' : [8, 16, 32, 64, 100, 300]
                f_k = self.image_size / self.steps[k]
                
                # デフォルトボックスの中心座標 x, y ただし 0~1 で規格化している
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                # アスペクト比 1 の小さいデフォルトボックス [cx, cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                
                # アスペクト比 1 の大きいデフォルトボックス [cx, cy, width, height]
                # 'min_sizes': [45, 99, 153, 207, 261, 315]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                
                # その他のアスペクト比の defBox [cx, cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
                
        # デフォルトボックスをテンソルに変換 torch.Size([8372, 4])
        output = torch.Tensor(mean).view(-1, 4)
        # デフォルトボックスが画像外にはみ出すのを防ぐため大きさを最大1，最小0にする
        output.clamp_(max=1, min=0)
        
        return output


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train or inferenceを指定
        self.num_classes = cfg["num_classes"]  # クラス数=21

        # SSDのネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 推論時はクラス「Detect」を用意します
        if phase == 'inference':
            self.detect = Detect()