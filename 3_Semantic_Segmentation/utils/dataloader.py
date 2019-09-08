#!/usr/bin/env python
# coding: utf-8

import os
import torch.utils.data as data
from PIL import Image
from .data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor

def make_datapath_list(rootpath):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス
    
    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    img_path_template = os.path.join(rootpath, "JPEGImages", "%s.jpg")
    anno_path_template = os.path.join(rootpath, "SegmentationClass", "%s.png")
    
    # 訓練と検証それぞれのファイル名を取得
    train_id_names = os.path.join(rootpath, "ImageSets/Segmentation/train.txt")
    val_id_names = os.path.join(rootpath, "ImageSets/Segmentation/val.txt")
    
    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()
    for line in open(train_id_names):
        file_id = line.strip()                     # 空白スペースと改行を除去
        img_path = (img_path_template % file_id)   #画像のパス
        anno_path = (anno_path_template % file_id) # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()
    for line in open(val_id_names):
        file_id = line.strip()                     # 空白スペースと改行を除去
        img_path = (img_path_template % file_id)   #画像のパス
        anno_path = (anno_path_template % file_id) # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list


# データの処理を行うクラス
class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする
    画像のサイズを input_size x input_size にし，訓練時はデータオーギュメンテーションする。

    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ
    color_mean : (R, G, B)
        各色チャネルの平均値
    color_std : (R, G, B)
        各色チャネルの標準偏差
    """
    
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose({
                Scale(scale=[0.5, 1.5]),                # 画像の拡大・縮小
                RandomRotation(angle=[-10, 10]),        # 回転
                RandomMirror(),                         # 左右の反転
                Resize(input_size),                     # input_size にリサイズ
                Normalize_Tensor(color_mean, color_std) # 色情報の標準化・テンソル化
            }),
            "val": Compose({
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            })
        }
        
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)
    

# Dataset クラス
class VOCDataset(data.Dataset):
    """
    VOC2012 の Dataset を作成するクラス，PyTorch の Dataset クラスを継承

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """
    
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        
    
    def __len__(self):
        ''' 画像の枚数を返す '''
        return len(self.img_list)
    
    
    def __getitem__(self, index):
        ''' 前処理した画像の Tensor 形式のデータとアノテーションを取得 '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img
    
    
    def pull_item(self, index):
        ''' 画像の Tensor 形式のデータ、アノテーションを取得する '''
        
        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)  # [高さ][幅][色 RGB]
        
        # 2. アノテーション画像の読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path) # [高さ][幅]
        
        # 3. 前処理の実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
        
        return img, anno_class_img


# (RGB) の色の平均値と標準偏差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

# データセットの作成
train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",
                          transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                         transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

# データの取り出し
print(val_dataset.__getitem__(0)[0].shape)
print(val_dataset.__getitem__(0)[1].shape)
print(val_dataset.__getitem__(0))

batch_size = 8
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}