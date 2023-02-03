# DCNVSS
Dual Correlation Network for Efficient Video Semantic Segmentation

This repository is the official implementation of "Dual Correlation Network for Efficient Video Semantic Segmentation” ( This paper is under submission, we will show it later)

## Install & Requirements
Requirements: `PyTorch >= 1.4.0, CUDA >= 10.0, and Python==3.8`

**To Install weightingFunction**
```
cd $DCNVSS_ROOT/Local-Attention-master
python setup.py build
```

**To Install Correlation**
```
cd $DCNVSS_ROOT/correlation
python setup.py build
```
## Usage
### Data preparation
Please follow [Cityscapes](https://www.cityscapes-dataset.com/) to download Cityscapes dataset. After correctly downloading, the file system is as follows:
````bash
$DCNVSS_ROOT/data
├── Cityscapes_video
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
````
### Training

1. Download pretrained PSP101 models [BaiduYun(Access Code:ghk4)]( https://pan.baidu.com/s/199rZZdlOhBt3ZiKbnBmCmQ) on Cityscapes dataset, and put them in a folder `./ckpt`.

2. Training requires 4 Nvidia GPUs.
````bash
# training Dual Correlation Network
bash ./train.sh
# training key frame selection module
bash ./train_KDM.sh
````
### Test
1. Download the trained weights from [BaiduYun(Access Code:bay9)]( https://pan.baidu.com/s/1Bf8Bc2KE_xO1hR6NCV1Wxw) and put them in a folder `./ckpt`.

2. Run the following commands:
````bash
bash ./eval_multipro.sh
````
## Acknowledgement
The code is heavily based on the following repositories:
- https://github.com/CSAILVision/sceneparsing
- https://github.com/zzd1992/Image-Local-Attention
- https://github.com/WeilunWang/correlation-layer

Thanks for their amazing works.

## Citation
We will show it later.

