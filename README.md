BARN is based on the code platform of **[mmaction2](https://github.com/open-mmlab/mmaction2) v0.22.0**(released in 05/03/2022). 



https://user-images.githubusercontent.com/53562283/227199696-adcc9f1b-7dfb-42e4-ae8b-cee1b78e590a.mp4



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


2- Please download the videos of socially housed macaques from [data](https://drive.google.com/file/d/1js0rWrHFzt_7M6A3w2eD9NBVJjhH6QBM/view?usp=sharing), then extract the video frames with the [ExtractFrame code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey/extract_monkey_frames.sh) and save them to the path of "BARN-monkey/data/monkey/frames/". These frames contain the whole frames in valuation set.

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


## License

This project is released under the [Apache 2.0 license](LICENSE).
