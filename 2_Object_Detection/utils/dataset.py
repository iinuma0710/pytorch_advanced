import os.path as osp
import numpy as np
import cv2
import random
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from .data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_datapath_list(rootpath):
    """
    データへのパスを格納したリストを作成
    
    Parameters
    ----------
    rootpath: str
        データフォルダへのパス
    
    Returns
    -------
    ret: train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
    
    # 訓練と検証についてそれぞれのファイル名を取得する
    train_id_names = osp.join(rootpath, 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath, 'ImageSets/Main/val.txt')
    
    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
    
    return train_img_list, train_anno_list, val_img_list, val_anno_list


class Anno_xml2list():
    """
    各画像のアノテーションデータを画像サイズで規格化しリスト形式に変換
    
    Attributes
    ----------
    classes: list
        VOC のクラス名を格納したリスト
    """
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        """
        Parameters
        ----------
        xml_path: str
            xml ファイルへのパス
        width: int
            対象画像の幅
        height: int
            対象画像の高さ
            
        Returns
        -------
        ret: [[xmin, ymin, xmax, ymax, label_idx], ...]
            物体のアノテーションデータを格納したリストで長さは画像内の物体数
        """
        
        # このリストに画像内のすべての物体のアノテーションを格納する
        ret = []
        
        # xml ファイルの読み込み
        xml = ET.parse(xml_path).getroot()
        
        # 画像内にある物体の数だけループ
        for obj in xml.iter('object'):
            # 検知が difficult となっているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
                
            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []
            
            name = obj.find('name').text.lower().strip() # 物体名を抽出
            bbox = obj.find('bndbox') # バウンディングボックスの情報
            
            # バウンディングボックスの情報を0~1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for pt in (pts):
                # 原点を (0, 0) にする
                cur_pixel = int(bbox.find(pt).text) - 1
                
                # 幅、高さで規格化
                if pt == 'xmin' or pt == 'xmax':  # x方向のときは幅で割算
                    cur_pixel /= width
                else:  # y方向のときは高さで割算
                    cur_pixel /= height

                bndbox.append(cur_pixel)
                
            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # resに[xmin, ymin, xmax, ymax, label_ind]を足す
            ret += [bndbox]

        return np.array(ret)
    

# DataTransform の実装
class DataTransform():
    """
    画像とアノテーションの前処理クラス
    訓練時と推論時で異なる動作をする
    訓練時はオーギュメンテーションを行う
    
    Attributes
    ----------
    input_size : int
        リサイズ後の画像サイズ
    color_mean : (B, G, R)
        各色チャネルの平均値
    """
    
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),        # int => float32 に変換
                ToAbsoluteCoords(),       # アノテーションデータの規格化を戻す
                PhotometricDistort(),     # 画像の色調などをランダムに変化させる
                Expand(color_mean),       # 画像のキャンパスを拡げる
                RandomSampleCrop(),       # 画像内の一部をランダムに切り出す
                RandomMirror(),           # 画像の反転を行う
                ToPercentCoords(),        # アノテーションデータを 0~1 に規格化
                Resize(input_size),       # 画像サイズを一辺 input_size にリサイズする
                SubtractMeans(color_mean) # BGR の色の平均値を引く
            ]),
            'val': Compose([
                ConvertFromInts(),        # int => float32 に変換
                Resize(input_size),       # 画像サイズを一辺 input_size にリサイズする
                SubtractMeans(color_mean) # BGR の色の平均値を引く
            ])
        }
        
    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定
        """
        return self.data_transform[phase](img, boxes, labels)
    

class VOCDataset(data.Dataset):
    """
    VOC2012 の Dataset を作成するクラス
    PyTorch の Dataset クラスを継承する
    
    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'val'
        訓練か推論かを指定する
    transform : object
        前処理クラスのインスタンス
    transform_anno : object
        xml のアノテーションをリストに変換するインスタンス
    """
    
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno
        
    def __len__(self):
        '''画像枚数を返す'''
        return len(self.img_list)
    
    def __getitem__(self, index):
        '''前処理済みの画像のテンソル形式でのデータとアノテーションを取得'''
        im, gt, h, w = self.pull_item(index)
        return im, gt
    
    def pull_item(self, index):
        '''前処理済みの画像のテンソル形式でのデータとアノテーション，画像の高さと幅を取得'''
        # 画像を読み込む
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channel = img.shape

        # アノテーションをリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)
        
        # 前処理の実行
        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4])
        
        # 色チャネルを BGR => RGB に変更，(高さ，幅，色チャネル) を (色チャネル，高さ，幅) に変換
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        
        # バウンディングんボックスとラベルをセットにして ndarray を作成
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, gt, height, width
    
    
def make_VOC_dataset():
    color_mean = (104, 117, 123)
    input_size = 300
    # 訓練用・検証用データのリストアップ
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
    # VOC データセットのクラス一覧
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    # 訓練データセット
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", 
                               transform=DataTransform(input_size, color_mean),
                               transform_anno=Anno_xml2list(voc_classes))
    # 検証データセット
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                             transform=DataTransform(input_size, color_mean),
                             transform_anno=Anno_xml2list(voc_classes))
    
    return train_dataset, val_dataset