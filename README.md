# [VoV3D](https://arxiv.org/abs/2012.00317)

[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/pdf/2012.00317.pdf)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diverse-temporal-aggregation-and-depthwise/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=diverse-temporal-aggregation-and-depthwise) \
VoV3D is an efficient and effective 3D backbone network for temporal modeling implemented on top of [PySlowFast](https://github.com/facebookresearch/SlowFast).



> **[Diverse Temporal Aggregation and Depthwise Spatiotemporal Factorization for Efficient Video Classification](https://arxiv.org/abs/2012.00317)**<br>
> [Youngwan Lee](https://github.com/youngwanLEE), Hyung-Il Kim, Kimin Yun, and Jinyoung Moon<br>
> Electronics and Telecommunications Research Institute ([ETRI](https://www.etri.re.kr/eng/main/main.etri))<br>
> pre-print : https://arxiv.org/abs/2012.00317

<div align="center">
  <img src="https://dl.dropbox.com/s/v5v8in1x0womk2a/github_figure.jpg" width="850px" />
</div>



## Abstract
Video classification researches that have recently attracted attention are the fields of temporal modeling and 3D efficient architecture. However, the temporal modeling methods are not efficient or the 3D efficient architecture is less interested in temporal modeling. For bridging the gap between them, we propose an efficient temporal modeling 3D architecture, called **VoV3D**, that consists of a temporal one-shot aggregation (**T-OSA**) module and depthwise factorized component, **D(2+1)D**. The T-OSA is devised to build a feature hierarchy by aggregating temporal features with different temporal receptive fields. Stacking this T-OSA enables the network itself to model short-range as well as long-range temporal relationships across frames without any external modules. Inspired by kernel factorization and channel factorization, we also design a depthwise spatiotemporal factorization module, named, D(2+1)D that decomposes a 3D depthwise convolution into two spatial and temporal depthwise convolutions for making our network more lightweight and efficient. By using the proposed temporal modeling method (T-OSA), and the efficient factorized component (D(2+1)D), we construct two types of VoV3D networks, VoV3D-M and VoV3D-L. Thanks to its efficiency and effectiveness of temporal modeling, VoV3D-L has 6x fewer model parameters and 16x less computation, surpassing a state-of-the-art temporal modeling method on both Something-Something and Kinetics-400. Furthermore, VoV3D shows better temporal modeling ability than a state-of-the-art efficient 3D architecture, X3D having comparable model capacity. We hope that VoV3D can serve as a baseline for efficient video classification.


## Main Result
Our results (X3D & VoV3D) are trained in the same environment.
 - V100 8 GPU machine
 - same training protocols (BASE_LR, LR_POLICY, batch size, etc)
 - pytorch 1.6
 - CUDA 10.1

 
*Please refer to our [paper](https://arxiv.org/abs/2012.00317) or [configs files](configs) for the details.\
*When you want to reproduce the same results, you just train the model with [configs](configs) on the **8** GPU machine.
If you change `NUM_GPUS` or `TRAIN.BATCH_SIZE` values, you have to adjust `BASE_LR`. \
*IM and K-400 denote ImageNet and Kinetics-400, respectively.
### Something-Something-V1

| Model                                                                                       | Backbone | Pretrain | #Frame | Param. |  GFLOPs |   Top-1  |   Top-5  |                                           weight                                           |
|---------------------------------------------------------------------------------------------|:--------:|:--------:|:------:|:------:|:-------:|:--------:|:--------:|:------------------------------------------------------------------------------------------:|
| [TSM](https://github.com/mit-han-lab/temporal-shift-module)                                 |   R-50   |   K-400  |   16   |  24.3M |   33x6  |   48.3   |   78.1   |    [`link`](https://github.com/mit-han-lab/temporal-shift-module#something-something-v1)   |
| TSM+[TPN](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md#something-something) |   R-50   |    IM    |    8   |   N/A  |   N/A   |   50.7   |     -    | [`link`](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md#something-something) |
| [TEA](https://github.com/Phoenix1327/tea-action-recognition)                                |   R-50   |    IM    |   16   |  24.4M |  70x30  |   52.3   |   81.9   |                                              -                                             |
| [ip-CSN-152](https://github.com/facebookresearch/VMZ/tree/master/c2)                        |     -    |     -    |   32   |  29.7M | 74.0x10 |   49.3   |     -    |                                              -                                             |
| [X3D](configs/SSv1/x3d/x3d_M.yaml)                                                          |     M    |     -    |   16   |  3.3M  |  6.1x6  |   46.4   |   75.3   |         [`link`](https://dl.dropbox.com/s/lb9uvj4tl19xzmn/x3d_M_ssv1.pth)                  |
| **VoV3D**                                                                                   |     M    |     -    |   16   |  3.3M  |  5.7x6  |   48.1   |   76.9   |         [`link`](https://dl.dropbox.com/s/usbcudmhpknqvtd/vov3d_M_f16_ssv1.pth)            |
| **VoV3D**                                                                                   |     M    |     -    |   32   |  3.3M  |  11.5x6 |   49.8   |   78.0   |         [`link`](https://dl.dropbox.com/s/fvet69zyeswe25v/vov3d_M_f32_ssv1.pth)            |
| **VoV3D**                                                                                   |     M    |   K-400  |   32   |  3.3M  |  11.5x6 |   52.6   |   80.4   |         [`link`](https://etri.gov-dooray.com/share/drive-files/kdhawuidbrbd.C59lPfB2Toul7wcFMk1yeQ)          |
| [X3D](configs/SSv1/x3d/x3d_L.yaml)                                                          |     L    |     -    |   16   |  5.6M  |  9.1x6  |   47.0   |   76.4   |         [`link`](https://dl.dropbox.com/s/yixwwfv6mv2bpnd/x3d_L_ssv1.pth)                  |
| **VoV3D**                                                                                   |     L    |     -    |   16   |  5.8M  |  9.3x6  |   49.5   |   78.0   |         [`link`](https://dl.dropbox.com/s/bdjam5tyaedkczl/vov3d_L_f16_ssv1.pth)            |
| **VoV3D**                                                                                   |     L    |     -    |   32   |  5.8M  |  20.9x6 |   50.6   |   78.7   |         [`link`](https://dl.dropbox.com/s/g6lc5sj8h7f2r5v/vov3d_L_f32_ssv1.pth)            |
| **VoV3D**                                                                                   |     L    |   K-400  |   32   |  5.8M  |  20.9x6 | **54.9** | **82.3** |         [`link`](https://etri.gov-dooray.com/share/drive-files/kdhawuidbrbd.r2fJ4HSDSWmFYNo1ARZuHQ)          |


### Something-Something-V2

| Model                                                                                                         | Backbone | Pretrain | #Frame | Param. |  GFLOPs |   Top-1  |   Top-5  |                                                                weight                                                               |
|---------------------------------------------------------------------------------------------------------------|:--------:|:--------:|:------:|:------:|:-------:|:--------:|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
| [TSM](https://github.com/mit-han-lab/temporal-shift-module)                                                   |   R-50   |   K-400  |   16   |  24.3M |   33x6  |   63.0   |   88.1   |                        [`link`](https://github.com/mit-han-lab/temporal-shift-module#something-something-v2)                        |
| TSM+[TPN](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md#something-something)                   |   R-50   |    IM    |    8   |   N/A  |   N/A   |   64.7   |     -    |                      [`link`](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md#something-something)                     |
| [TEA](https://github.com/Phoenix1327/tea-action-recognition)                                                  |   R-50   |    IM    |   16   |  24.4M |  70x30  |   65.1   |   89.9   |                                             -                                                                                       |
| [SlowFast 16x8](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md#something-something-v2) |   R-50   |   K-400  |   64   |  34.0M | 131.4x6 |   63.9   |   88.2   | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/pyslowfast/model_zoo/multigrid/model_zoo/Kinetics/SLOWFAST_8x8_R50_stepwise.pkl) |
| [X3D](configs/SSv2/x3d/x3d_M.yaml)                                                                            |     M    |     -    |   16   |  3.3M  |  6.1x6  |   63.0   |   87.9   |                              [`link`](https://dl.dropbox.com/s/b6gjywzrn85sa89/x3d_M_ssv2.pth)                                      |
| **VoV3D**                                                                                                     |     M    |     -    |   16   |  3.3M  |  5.7x6  |   63.2   |   88.2   |                              [`link`](https://dl.dropbox.com/s/hz34ltdl6iukdxc/vov3d_M_f16_ssv2.pth)                                |
| **VoV3D**                                                                                                     |     M    |     -    |   32   |  3.3M  |  11.5x6 |   64.2   |   88.8   |                              [`link`](https://dl.dropbox.com/s/z6qtyjgf0v13jnq/vov3d_M_f32_ssv2.pth)                                |
| **VoV3D**                                                                                                     |     M    |   K-400  |   32   |  3.3M  |  11.5x6 |   65.2   |   89.4   |                              [`link`](https://etri.gov-dooray.com/share/drive-files/kdhawuidbrbd.PHQw3KwbRI-lIcopWjm-ew)            |
| [X3D](configs/SSv2/x3d/x3d_L.yaml)                                                                            |     L    |     -    |   16   |  5.6M  |  9.1x6  |   62.7   |   87.7   |                              [`link`](https://dl.dropbox.com/s/iaqwbejrkbd7372/x3d_L_ssv2.pth)                                      |
| **VoV3D**                                                                                                     |     L    |     -    |   16   |  5.8M  |  9.3x6  |   64.1   |   88.6   |                              [`link`](https://dl.dropbox.com/s/678efl8zyrmq7ch/vov3d_L_f16_ssv2.pth)                                |
| **VoV3D**                                                                                                     |     L    |     -    |   32   |  5.8M  |  20.9x6 |   65.8   |   89.5   |                              [`link`](https://dl.dropbox.com/s/f4330e5htdiwqa5/vov3d_L_f32_ssv2.pth)                                |
| **VoV3D**                                                                                                     |     L    |   K-400  |   32   |  5.8M  |  20.9x6 | **67.3** | **90.5** |                              [`link`](https://etri.gov-dooray.com/share/drive-files/kdhawuidbrbd.pEg57hjpSkOp2boe1oNZvQ)                            |

### Kinetics-400

| Model                                                                                                             | Backbone | Pretrain | #Frame | Param. |  GFLOPs | Top-1 | Top-5 |                                             weight                                             |
|-------------------------------------------------------------------------------------------------------------------|:--------:|:--------:|:------:|:------:|:-------:|:-----:|:-----:|:----------------------------------------------------------------------------------------------:|
| [X3D (PySlowFast, 300e)](https://github.com/facebookresearch/SlowFast/blob/master/configs/Kinetics/X3D_M.yaml)    |     M    |     -    |   16   |  3.8M  |  6.2x30 |  76.0 |  92.3 |           [`link`](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth)            |
| [X3D (our, 256e)](configs/Kinetics/x3d/X3D_M_256e.yaml)                                                           |     M    |     -    |   16   |  3.8M  |  6.2x30 |  75.0 |  92.1 |           [`link`](https://dl.dropbox.com/s/s87lnn7ch4a6e47/x3d_M_256e_k400.pth)               |
| **VoV3D**                                                                                                         |     M    |     -    |   16   |  3.8M  |  4.4x30 |  73.9 |  91.6 |           [`link`](https://dl.dropbox.com/s/hpk172nbvm3w5p1/vov3d_M_k400.pth)                  |
| [X3D (PySlowfast)](https://github.com/facebookresearch/SlowFast/blob/master/configs/Kinetics/X3D_L.yaml)          |     L    |     -    |   16   |  6.1M  | 24.8x30 |  77.5 |  92.9 |           [`link`](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_l.pyth)            |
| **VoV3D**                                                                                                         |     L    |     -    |   16   |  6.2M  |  9.3x30 |  76.3 |  92.9 |           [`link`](https://dl.dropbox.com/s/lzmq8d4dqyj8fj6/vov3d_L_k400.pth)                  |

*We note that since X3D-M (PySlowFast) was trained for 300 epochs, we re-train the X3D-M (our, 256e) with the same 256 epochs with VoV3D-M.

## Installation & Data Preparation
Please refer to [INSTALL.md](INSTALL.md) for installation and [DATA.md](DATA.md) for data preparation. \
**Important** : We used [depthwise 3D Conv pytorch patch](https://github.com/pytorch/pytorch/pull/40801) for accelearating GPU runtime.

## Training & Evaluation
We provide brief examples for getting started. If you want to know more details, please refer to [instruction](
) of PySlowFast.

### Training
#### from scratch
 - VoV3D-L on Kinetics-400

```shell
python tools/run_net.py \
  --cfg configs/Kinetics/vov3d/vov3d_L.yaml \
  DATA.PATH_TO_DATA_DIR path/to/your/kinetics \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 64
```

You can also designate each argument in the config file. If you want to train with our default setting (e.g., 8GPUs, 64 batch size, etc), you just use this command. (Set `DATA.PATH_TO_DATA_DIR` with your real data path)

```shell
python tools/run_net.py --cfg configs/Kinetics/vov3d/vov3d_L.yaml
```

 - VoV3D-L on Something-Something-V1

```shell
python tools/run_net.py \
  --cfg configs/SSv1/vov3d/vov3d_L_F16.yaml \
  DATA.PATH_TO_DATA_DIR path/to/your/ssv1 \ 
  DATA.PATH_PREFIX path/to/your/ssv1
``` 

#### Finetuning by using Kinetics-400 pretrained weight.
First, you have to download the weights pretrained on Kinetics-400.

 - [vov3d_M_k400_weight](https://dl.dropbox.com/s/hpk172nbvm3w5p1/vov3d_M_k400.pth)
 - [vov3d_L_k400_weight](https://dl.dropbox.com/s/lzmq8d4dqyj8fj6/vov3d_L_k400.pth)

One thing you should keep in mind is that `TRAIN.CHECKPOINT_FILE_PATH` is the downloaded weight.

For Something-Something-V2,

```shell
cd VoV3D
mkdir -p output/pretrained
wget https://dl.dropbox.com/s/lzmq8d4dqyj8fj6/vov3d_L_k400.pth

python tools/run_net.py \
  --cfg configs/SSv2/vov3d/finetune/vov3d_L_F16.yaml \
  TRAIN.CHECKPOINT_FILE_PATH path/to/the/pretrained/vov3d_L_k400.pth \
  DATA.PATH_TO_DATA_DIR path/to/your/ssv2 \
  DATA.PATH_PREFIX path/to/your/ssv2
```

### Testing

When testing, you have to set `TRAIN.ENABLE` to `False` and `TEST.CHECKPOINT_FILE_PATH` to `path/to/your/checkpoint`.

```shell
python tools/run_net.py \
  --cfg configs/Kinetics/vov3d/vov3d_L.yaml \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint
```

If you want to test with single clip and single-crop, set `TEST.NUM_ENSEMBLE_VIEWS` and `TEST.NUM_SPATIAL_CROPS` to 1, respectively.

```shell
python tools/run_net.py \
  --cfg configs/Kinetics/vov3d/vov3d_L.yaml \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1
```
 
For Kinetics-400, 30-views : `TEST.NUM_ENSEMBLE_VIEWS` 10 &  `TEST.NUM_SPATIAL_CROPS` 3 \
For Something-Something, 6-views : `TEST.NUM_ENSEMBLE_VIEWS` 2 &  `TEST.NUM_SPATIAL_CROPS` 3
   

## License
The code and the models in this repo are released under the [CC-BY-NC4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/legalcode). See the [LICENSE](LICENSE.md) file.


## <a name="CitingVoV3D"></a>Citing VoV3D

```BibTeX
@article{lee2020vov3d,
  title={Diverse Temporal Aggregation and Depthwise Spatiotemporal Factorization for Efficient Video Classification},
  author={Lee, Youngwan and Kim, Hyung-Il and Yun, Kimin and Moon, Jinyoung},
  journal={arXiv preprint arXiv:2012.00317},
  year={2020}
}

@inproceedings{lee2019energy,
  title = {An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection},
  author = {Lee, Youngwan and Hwang, Joong-won and Lee, Sangrok and Bae, Yuseok and Park, Jongyoul},
  booktitle = {CVPR Workshop},
  year = {2019}
}

@inproceedings{lee2020centermask,
  title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
  author={Lee, Youngwan and Park, Jongyoul},
  booktitle={CVPR},
  year={2020}
}
```

## Acknowledgement
We appreciate developers of [PySlowFast](https://github.com/facebookresearch/SlowFast) for such wonderful framework. \
This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. B0101-15-0266, Development of High Performance Visual BigData Discovery Platform for Large-Scale Realtime Data Analysis and No. 2020-0-00004, Development of Previsional Intelligence based on Long-term Visual Memory Network).
