# pytorch_advanced
つくりながら学ぶ！PyTorchによる発展ディープラーニングのリポジトリです．

## 環境設定
ローカルのGPU環境で実行するための環境を整備します．

### 手元の環境
ノートPCとデスクトップPCの環境がありますが両方とも同じように環境設定できます．

- ノートPC
  - CPU: Intel Core i7-8565U @ 1.8GHz × 8
  - GPU: GeForce MX150
  - メモリ: 16GB

- デスクトップPC（これから作ります）
  - CPU: AMD Ryzen 9 3950X
  - GPU: GeForce GTX1080Ti
  - メモリ: 64GB
 
- OS: Ubuntu 18.04.2 LTS

### ソフトウェア要件
別件で TensorFlow も使いたいので両方が動作する環境は次のようになっています．
- Python 2.7 or Python 3.5 以上
- nVidia Driver 410.x 以降
- CUDA 9.0 or CUDA 10.0
- cuDNN 7.4.1 以降

### GPU ドライバのインストール
最初にデフォルトのドライバを無効にします．

```bash
$ sudo vi /etc/modprobe.d/blacklist-nouveau.conf
  
  blacklist nouveau
  options nouveau modeset=0 
  
$ sudo update-initramfs -u
$ reboot
```
[PyTorch のページ](https://pytorch.org/)でドライバの対応状況を確認します．
とりあえず最新の安定版を入れておけば問題ないと思います．（2019/8/2 時点では 430.40）

```bash
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get update
$ sudo apt-get install nvidia-430.40
$ sudo apt-get install mesa-common-dev
$ sudo apt-get install freeglut3-dev
```

ここまで済んだら一度再起動して nvidia-smi で確認します．

### CUDA 10.0 の導入
TensorFlow も PyTorch もサポートは CUDA 10.0 だけなので，CUDA 10.1 は使いません．
（一応，PyTorch は CUDA 10.1 でも動きました．）
[CUDA 10.0 のダウンロードページ](/developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)から .deb ファイルをダウンロードしておきます．  
３行目で CUDA のバージョンをちゃんと指定しないと勝手に最新版の 10.1 が入ってしまうので注意．

```bash
$ sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
$ sudo apt update
$ sudo apt install cuda-10-0 cuda-drivers
$ sudo reboot
```

インストールできたら PATH を通します．
```bash
$ echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
$ sudo reboot
```

再起動して次のような表示が出てきたら成功．
```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

### cuDNN の導入
[cuDNN のダウンロードページ](https://developer.nvidia.com/rdp/cudnn-download)から以下の３つのファイルをダウンロードします．
- cuDNN v7.6.1 Runtime Library for Ubuntu18.04 (Deb)
- cuDNN v7.6.1 Developer Library for Ubuntu18.04 (Deb)
- cuDNN v7.6.1 Code Samples and User Guide for Ubuntu18.04 (Deb)

```bash
$ sudo dpkg -i libcudnn7_7.6.1.34-1+cuda10.0_amd64.deb
$ sudo dpkg -i libcudnn7-dev_7.6.1.34-1+cuda10.0_amd64.deb
$ sudo dpkg -i libcudnn7-doc_7.6.1.34-1+cuda10.0_amd64.deb
```

### PyTorch のインストール
pip を使えば一発でインストールできます．
```bash
$ pip install torch torchvision
```

以下のコマンドでGPUが認識されているか確認できます．
```python
import torch
torch.cuda.is_available()     # GPU が使えるなら True が返ってくる
torch.cuda.get_device_name(0) # GPU の機種が表示される
```
