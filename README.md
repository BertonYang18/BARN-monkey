BARN is based on the code platform of **[mmaction2](https://github.com/open-mmlab/mmaction2) v0.22.0**(released in 05/03/2022). 


![monkey_BARN_demo](https://user-images.githubusercontent.com/53562283/227211342-cdb51882-9316-491a-9c8e-ee8bf0be248f.gif)
<div align="center">
  <p style="font-size:1.5vw;">Behavior Detection Results of BARN on the macaque behavior dataset using the ground-truth bbox</p>
  </div>


## Note

1- Please use the files [mmcv](https://github.com/BertonYang18/BARN-monkey/tree/main/mmcv) and [mmdet](https://github.com/BertonYang18/BARN-monkey/tree/main/mmdet) in BARN instead of the original version.

2- Please download [ext.cpython-38-x86_64-linux-gnu.so](https://drive.google.com/file/d/1fgIpmBsJdoVywg6Ax_8QenBmSFdi2nA0/view?usp=sharing) and save it to the path of "BARN-monkey/mmcv/_ext.cpython-38-x86_64-linux-gnu.so".


## Data Preparation

1- Please download the ckpt files as follows.

|config | resolution | gpus | backbone |pretrain| val mAP(%) | gpu_mem(MB) | ckpt |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|mix_r50_4x16x1_20e_ava_rgb_custom |short-side 256|1x3| SlowFast | ava v2.1 | 64.3 | 24220 | [ckpt](https://drive.google.com/file/d/1AFP8iwZMhV3859XfxjHd04FL_O-KoMr8/view?usp=share_link) |
|mix_r50_4x16x1_20e_ava_rgb_custom_acrn |short-side 256|1x3| SlowFast | ava v2.1 | 59.3 | 24220 | [ckpt](https://drive.google.com/file/d/14VpkKnzU_GGp7510d0ZTpJU-2KsKuhGI/view?usp=share_link) |
|mix_slowonly_r50_4x16x1 |short-side 256|1x3| SlowOnly | ava v2.1 | 59.4 | 24220 | [ckpt](https://drive.google.com/file/d/1GKspAwQL6APx7EBg3jy_dJ9GqqL_9Hz4/view?usp=share_link) |
|mix_r50_4x16x1_20e_ava_rgb_custom_baseline |short-side 256|1x3| SlowFast | ava v2.1 | 58.8 | 24220 | [ckpt](https://drive.google.com/file/d/1ugIGRbFjhq19hdYXaTNvwjERxPbXtaUb/view?usp=share_link) |


2- Please download the videos of socially housed macaques from [data](https://drive.google.com/drive/folders/1OnbBmgrOv0lLPgof0dVh3f4VsdNx_RSj?usp=share_link), then extract the video frames with the [ExtractFrame code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey/extract_monkey_frames.sh) and save them to the path of "BARN-monkey/data/monkey/frames/". These frames contain the whole frames in valuation set.

## Training and testing

You can use the file "tools/train_monkey.py" for training and use the file "tools/test_monkey.py" for inference.

## Original readme.md of mmaction2
The followings are the content in readme.md of mmaction2:

[📘Documentation](https://mmaction2.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmaction2.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmaction2.readthedocs.io/en/latest/modelzoo.html) |
[🆕Update News](https://mmaction2.readthedocs.io/en/latest/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmaction2/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmaction2/issues/new/choose)

## Introduction

English | [简体中文](/README_zh-CN.md)

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)


## Installation

Please refer to [install.md](docs/install.md) for installation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction2.
There are also tutorials:

- [learn about configs](docs/tutorials/1_config.md)
- [finetuning models](docs/tutorials/2_finetune.md)
- [adding new dataset](docs/tutorials/3_new_dataset.md)
- [designing data pipeline](docs/tutorials/4_data_pipeline.md)
- [adding new modules](docs/tutorials/5_new_modules.md)
- [exporting model to onnx](docs/tutorials/6_export_model.md)
- [customizing runtime settings](docs/tutorials/7_customize_runtime.md)

A Colab tutorial is also provided. You may preview the notebook [here](demo/mmaction2_tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb) on Colab.

## Monkey detector of BARN
The [monkey detector](https://github.com/BertonYang18/BARN_detector_YOLOv7) can generate the proposals including identities and bounding boxes of socially housed macaques.Specifically, using the monkey detector as follows:
1. Using the monkey detector to generate the proposals, which is the format of txt file.
2. Using the [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) to convert the txt results into pickle (.pkl) file.
3. Adopting the target config file and GPU in train_monkey.py / test_monkey.py 
4. Checking the path of .pkl file in the config file such as [BARN.py](https://github.com/BertonYang18/BARN-monkey/blob/main/configs/detection/monkey_interaction/mix_r50_4x16x1_20e_ava_rgb_custom.py).
After the above preparation, performing the following steps:
1- Running the train_monkey.py / test_monkey.py to generate the behavior predictions file (.csv).
2- Using the [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) to fuse the result of the monkey detector and the result of BARN into final behavior predictions (.csv file). One line of the csv file represents one of the prediction results of a bbox. (One bbox may corresponde to multiple simultaneous behaviors, as the AVA dataset does)
3.1- Using the final behavior predictions for behavior analysis. The [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) provides the codes to generate the movement distance, duration of behaviors.
3.2- The [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) provides the codes to visualize the results on imgs and concatnate them into visual videos, which could show the identity, bbox, behaviors and movement trajectory. (There may be some bugs and we will correct them as soon as possible)
Although the prediction process is automatical, it may be difficult for some researchers. Thus we may develop a Graphical User Interface (GUI) like the [DEEPLUBCUT](https://github.com/BertonYang18/DeepLabCut) for application in the future.


[videoname, timestamp of]
## License

This project is released under the [Apache 2.0 license](LICENSE).
