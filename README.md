# SA-Text: Simple but Accurate Detector for Text of Arbitrary Shapes

## Requirements
* Python 2.7
* PyTorch v0.4.1+
* pyclipper
* Polygon2
* OpenCV 3.4 (for c++ version pse)
* opencv-python 3.4

## Introduction

Regression with gaussian map to detect text accurate.
![img](figure/WX20200214-220416@2x.png)

### training

```
python train_ic15.py --arch resnet50 --batch_size 4 --root_dir /home/gengjiajia/Store/Text-Detection-Datasets/icpr_dataset/ 
```

### testing

```
python eval_sanet.py --root_dir /home/gengjiajia/code/OCR/PSENet/touxiang_pian/   --resume checkpoints/ic15_resnet50_bs_4_ep_600/checkpoint.pth.tar  --gpus 1
```

### visualization

![img](figure/img_4744.jpg)

## TODO

- Support curve text
- Adjust dliate kernel for larget text
- Post-progress by c++
- TensorRT support


