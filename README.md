# pytorch_advanced
つくりながら学ぶ！PyTorchによる発展ディープラーニングのリポジトリです．

## 環境設定
ローカルのGPU環境で実行するための環境を整備します．

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

### CUDA 10.1 の導入
[nVidia のページ](https://developer.nvidia.com/)で確認して最新版の CUDA を入れます．

```bash
$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
$ sudo apt update
$ sudo apt install cuda cuda-drivers
$ sudo reboot
$ rm cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
```

インストールできたら PATH を通します．
```bash
$ echo 'export PATH="/usr/local/cuda-9.0/bin:$PATH"' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
$ sudo reboot
```

再起動して次のような表示が出てきたら成功．
```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Apr_24_19:10:27_PDT_2019
Cuda compilation tools, release 10.1, V10.1.168
```

### cuDNN の導入
[cuDNN のダウンロードページ](https://developer.nvidia.com/rdp/cudnn-download)から以下の３つのファイルをダウンロードします．
- cuDNN v7.6.2 Runtime Library for Ubuntu18.04 (Deb)
- cuDNN v7.6.2 Developer Library for Ubuntu18.04 (Deb)
- cuDNN v7.6.2 Code Samples and User Guide for Ubuntu18.04 (Deb)

```bash
$ sudo dpkg -i libcudnn7_7.6.2.24-1+cuda10.1_amd64.deb
$ sudo dpkg -i libcudnn7-dev_7.6.2.24-1+cuda10.1_amd64.deb
$ sudo dpkg -i libcudnn7-doc_7.6.2.24-1+cuda10.1_amd64.deb
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