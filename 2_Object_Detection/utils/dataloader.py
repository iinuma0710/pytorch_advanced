import torch
import torch.utils.data as data
from .dataset import make_VOC_dataset

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def od_collate_fn(batch):
    """
    Dataset から取り出すデータのサイズが画像ごとに異なるため collate_fn をカスタマイズ
    関数名の od は Object Detection の略
    ミニバッチ分の画像が並んだリスト変数 batch にミニバッチ番号を指定する次元を先頭に追加する
    """
    
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0]) # sample[0] は画像
        targets.append(torch.FloatTensor(sample[1])) # sample[1] は Grand Truth
        
    # imgs はミニバッチサイズの3次元テンソルのリスト
    # 要素を ([3, 300, 300]) => ([batch_num, 3, 300, 300]) の4次元テンソルに変換
    imgs = torch.stack(imgs, dim=0)
    
    # targets はミニバッチサイズの Grand Truth のリスト
    # 画像内の物体数を n として，各要素は [n, 5] の行列
    # 5は [xmin, ymin, xmax, ymax, class_index]
    
    return imgs, targets


def make_VOC_dataloader():
    ''' VOC2012 のデータローダを作成する '''
    batch_size = 4
    train_dataset, val_dataset = make_VOC_dataset()
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    
    return dataloaders_dict