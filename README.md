# HiGS-Calib: A Hierarchical 3D Gaussian Splatting based Targetless Local-Consistent LiDAR-Camera Calibration Method

This repository contains the official authors implementation associated with the paper "HiGS-Calib: A Hierarchical 3D Gaussian Splatting based Targetless Local-Consistent LiDAR-Camera Calibration Method". 

### Testing Environment

We use Ubuntu 22.04 + CUDA 12.5 + PyTorch 2.4.0

### Installing Dependencies

#### 3DGS

Please first configure the environment according to the official configuration tutorial of [gaussian-splatting ](https://github.com/graphdeco-inria/gaussian-splatting).

#### ROMA

HiGS-Calib uses ROMA to extract inter-frame flows, please following the configuration tutorial of [ROMA](https://github.com/Parskatt/RoMa) to configure the environment.

To use HiGS, you need to download the weights of ROMA from the official implementation or our [link](https://drive.google.com/drive/folders/13Ftx61708xNP4JHke_9VBFOawWCZfcit?usp=sharing)

#### Pytorch3D

Installing pytorch3D, using command
```shell
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Dataset
The reorganized version of [KITTI360](https://www.cvlibs.net/datasets/kitti-360/) prepared for HiGS-Calib is [offered](https://drive.google.com/drive/folders/13Ftx61708xNP4JHke_9VBFOawWCZfcit?usp=sharing).

### Running

To run the calibrator, simply use

```shell
python calibrate.py -s <path_of_the_dataset> --flow_path <path_of_the_flow_model_weights>  --optimizer_type sparse_adam  --cam_id X --data_seq X
```
For example


```shell
python calibrate.py -s HiGS-Calib-dataset/KITTI360 --flow_path weights/ --optimizer_type sparse_adam  --cam_id 00 (00 or 01) --data_seq 2 (0-4)
```
