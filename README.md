# SA-Text: Simple but Accurate Detector for Text of Arbitrary Shapes

## Requirements
* Python 2.7
* PyTorch v0.4.1+
* pyclipper
* Polygon2
* opencv-python 3.4

## Introduction

Regression with gaussian map to detect text accurate.
<img src="figure/WX20200214-220416@2x.png" alt="img" style="zoom:50%;" />

### training

```shell
python train_ic15.py --arch resnet50 --batch_size 4 --root_dir $data_root_dir  
```

### testing

```shell
python eval_sanet.py --root_dir $data_root_dir  --resume checkpoints/ic15_resnet50_bs_4_ep_xxx/checkpoint.pth.tar  --gpus 1
```

### visualization

Training_data: MTWI dataset

|                                                              |                                                        |                                                        |
| ------------------------------------------------------------ | :----------------------------------------------------: | ------------------------------------------------------ |
| <img src="figure/1.png" alt="img" style="zoom:25%;" />| <img src="figure/2.png" alt="img" style="zoom:25%;" /> | <img src="figure/3.png" alt="img" style="zoom:25%;" /> |



## Differences from original paper

Here are two differences from paper: postprogress algorithm and outputs of network.

here are two outputs of networks: border_map and guassian map. **Border map** is used to seperate from two text instances, and **gaussian map** is used to generate text center region. For afraid of two text center region are detach, so we can use border map to delete these pixels that are in two instances border; then we use text center region to generate text instances, finally, we expand text instances by dilate in opencv.

