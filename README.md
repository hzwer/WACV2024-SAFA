# Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution
## [YouTube](https://youtu.be/J4N75OJPGJc) | [Poster](https://drive.google.com/file/d/1kiBWp-qP2lCRIRxbmfOAVtnjleRF0ICq/view?usp=share_link) | [Enhancement Model](https://github.com/hzwer/Practical-RIFE/blob/main/README.md#video-enhancement) | [demo](https://www.youtube.com/watch?v=QII2KQSBBwk) | [中文介绍](https://zhuanlan.zhihu.com/p/668775986)
## Introduction
We want to increase video resolution and frame rates end-to-end (end-to-end STVSR). This project is the implement of [Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution](http://arxiv.org/abs/2310.17294). Our SAFA network outperforms recent state-of-the-art methods such as TMNet and VideoINR by an average improvement of over 0.5dB on PSNR, while requiring less than half the number of parameters and only 1/3 computational costs. -> [author website](https://github.com/hzwer)

We have released some dedicated visual effect models for ordinary users. Some insights on multi-scale processing and feature fusion are reflected in RIFE applications, see [Practical-RIFE]([https://github.com/hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE#video-enhancement)). 

Space-Time Super-Resolution: 

![slomo_origin](https://github.com/megvii-research/WACV2024-SAFA/assets/10103856/aa9710a8-4b23-4c14-adaa-d864431faebd) -> ![slomo](https://github.com/megvii-research/WACV2024-SAFA/assets/10103856/58728e32-ca3b-4cc2-8b8f-b68a7ff9e2ee)

<img width="510" alt="image" src="https://github.com/megvii-research/WACV2024-SAFA/assets/10103856/a243c9e2-243e-4ce6-a5c0-3739d98eb22c">

## CLI Usage

### Installation

```
git clone git@github.com:megvii-research/WACV2024-SAFA.git
cd WACV2024-SAFA
pip3 install -r requirements.txt
```

Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1PCYRfKwMkymP0V5dmcmGwrKu0lU7xSZ0/view?usp=share_link).

### Run

**Image Interpolation**
```
python3 inference_img.py --img demo/i0.png demo/i1.png --exp=3
```
(2^3=8X interpolation results)

```
python3 inference_img.py --img demo/i0.png demo/i1.png --ratio=0.4
```
(for an arbitrary timestep)

## Training and Reproduction
We use 16 CPUs, 4 GPUs for training: 
```
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --world_size=4
```
The training scheme is mainly adopted from [RIFE](https://github.com/megvii-research/ECCV2022-RIFE).

![image](https://github.com/hzwer/WACV2024-SAFA/assets/10103856/be0c151d-f8a3-465c-ac78-45d688a31c70) ![image](https://github.com/hzwer/WACV2024-SAFA/assets/10103856/246ba74f-44fd-4001-8375-bb41e9adbcad)

## Recommend
We sincerely recommend some related papers:

ECCV22 - [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://github.com/megvii-research/ECCV2022-RIFE)

CVPR23 - [A Dynamic Multi-Scale Voxel Flow Network for Video Prediction](https://huxiaotaostasy.github.io/DMVFN/)

## Citation
If you think this project is helpful, please feel free to leave a star or cite our paper:

```
@inproceedings{huang2024safa,
  title={Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution},
  author={Huang, Zhewei and Huang, Ailin and Hu, Xiaotao and Hu, Chen and Xu, Jun and Zhou, Shuchang},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```
## Reference

[RIFE](https://github.com/megvii-research/ECCV2022-RIFE)   [DMVFN](https://huxiaotaostasy.github.io/DMVFN/)   [TMNet](https://github.com/CS-GangXu/TMNet)

[ZoomingSlomo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)    [VideoINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution)   

![image](https://github.com/megvii-research/WACV2024-SAFA/assets/10103856/d8b92072-bcf7-4d9d-bb27-26c07d85a154)

