BARN is based on the platform of **[mmaction2](https://github.com/open-mmlab/mmaction2) v0.22.0**(released in 05/03/2022). 

## Note

1- Using the file "mmcv" and "mmdet" in BARN instead of the original version.

2- Downloading "mmcv/_ext.cpython-38-x86_64-linux-gnu.so" from the original version and saving it in the file "mmcv" in BARN.


## Data Preparation

1- Please download the video frames of socially housed macaques from [ckpt](https://drive.google.com/file/d/1xDomKg2tJFlZzPDJZRcWA-92MKK0whcO/view?usp=sharing) and save them in the path of "BARN-monkey/data/monkey/frames/".

|config | resolution | gpus | backbone |pretrain| mAP(%) | gpu_mem(MB) | ckpt |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[mix_r50_4x16x1_20e_ava_rgb_custom](https://github.com/BertonYang18/BARN-monkey/tree/main/configs/detection/monkey) |short-side 256|1x3| SlowFast | ava v2.1 | 58.8 | 24220 | BARR_monkey_58.8.pth |
|[mix_acrn](https://github.com/BertonYang18/BARN-monkey/tree/main/configs/detection/monkey) |short-side 256|1x3| SlowFast | ava v2.1 | 56.97 | 24220 | ACRN_56.97.pth |
|[mix_slowonly_r50_4x16x1](https://github.com/BertonYang18/BARN-monkey/tree/main/configs/detection/monkey) |short-side 256|1x3| SlowOnly | ava v2.1 | 56.72 | 24220 | slowonly_56.72.pth |
|[mix_r50_4x16x1_20e_ava_rgb_custom_baseline](https://github.com/BertonYang18/BARN-monkey/tree/main/configs/detection/monkey) |short-side 256|1x3| SlowFast | ava v2.1 | 56.03 | 24220 | baseline_56.03.pth |


2- Please download the checkpoint files from (we will publish the link as soon as possible) and save them in the path of "BARN-monkey/checkpoint/".

## Training and testing

Please use the file "tools/train_monkey.py" for training and use the file "tools/test_monkey.py" for inference.

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


## License

This project is released under the [Apache 2.0 license](LICENSE).
