BARN is based on the code platform of **[mmaction2](https://github.com/open-mmlab/mmaction2) v0.22.0**(released in 05/03/2022). 
## Results show
![monkey_BARN_demo](https://user-images.githubusercontent.com/53562283/227211342-cdb51882-9316-491a-9c8e-ee8bf0be248f.gif)
<div align="center">
  <p style="font-size:1.5vw;">Behavior Detection Results of BARN on the macaque behavior dataset using the ground-truth bbox</p>
  <p style="font-size:1.5vw;">Visualization results of the bounding boxes, identity, movement trajectories and behaviors of macaques for two videos. The bounding box and movement trajectory of each macaque are drawn in the color of its collar. The identities and behaviors of macaques are separated by '#', and the simultaneous behavior is separated by '-'.</p>
  <p style="font-size:1.5vw;">identities including yellow, green, red, black, white (0~4)</p>
  <p style="font-size:1.5vw;">behaviors 0:sitting on the high shelf     9: walking     10: climbing.</p>
  </div>

See  the [order_number_of_behaviors_in_paper.doc](https://github.com/BertonYang18/BARN-monkey/blob/main/data) for the order number and definition of behaviors in the paper (0~18).

## Instruction videos for using the code
We made the [videos](https://pan.baidu.com/s/19kWRaqO7V_Ospyum29MsgA?pwd=euv3) (Extraction code:：euv3) to show how to use our code for detecting a new video. (We would add the captions to the videos as soon as possible.) Please prepare the data and code environment in advance as the following steps:


## Data Preparation

1- Please download the learned weights (ckpt files) on the proposed macaque behavior datasets:

|config | resolution | gpus | backbone |pretrain| val mAP(%) | gpu_mem(MB) | ckpt |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|mix_r50_4x16x1_20e_ava_rgb_custom |short-side 256|1x3| SlowFast | ava v2.1 | 64.3 | 24220 | [ckpt](https://drive.google.com/file/d/1AFP8iwZMhV3859XfxjHd04FL_O-KoMr8/view?usp=share_link) |
|mix_r50_4x16x1_20e_ava_rgb_custom_acrn |short-side 256|1x3| SlowFast | ava v2.1 | 59.3 | 24220 | [ckpt](https://drive.google.com/file/d/14VpkKnzU_GGp7510d0ZTpJU-2KsKuhGI/view?usp=share_link) |
|mix_slowonly_r50_4x16x1 |short-side 256|1x3| SlowOnly | ava v2.1 | 59.4 | 24220 | [ckpt](https://drive.google.com/file/d/1GKspAwQL6APx7EBg3jy_dJ9GqqL_9Hz4/view?usp=share_link) |
|mix_r50_4x16x1_20e_ava_rgb_custom_baseline |short-side 256|1x3| SlowFast | ava v2.1 | 58.8 | 24220 | [ckpt](https://drive.google.com/file/d/1ugIGRbFjhq19hdYXaTNvwjERxPbXtaUb/view?usp=share_link) |

2- Please download the videos of socially housed macaques from [all videos](https://pan.baidu.com/s/1lXfaw8aF9koXrOJHMGiMvQ?pwd=1582)(Extraction code: 1582) and [val&test videos](https://drive.google.com/drive/folders/1OnbBmgrOv0lLPgof0dVh3f4VsdNx_RSj?usp=share_link), then extract the video frames with the [ExtractFrame code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey/extract_monkey_frames.sh) and save them to the path of "BARN-monkey/data/monkey/frames/". These frames contain the whole frames of training, validation and test set.

3- Please download the [checkpoint files](https://drive.google.com/file/d/1dJa0r9ejwN-COswHeKXE-W4EuqrW4HVK/view?usp=share_link) and put them in a path such as 'checkpoint/'.
4- Remember to change the involved paths such as the paths of data in [config file](https://github.com/BertonYang18/BARN-monkey/blob/main/configs/detection/monkey_interaction/mix_r50_4x16x1_20e_ava_rgb_custom.py), the paths of checkpoints and weight in [train_monkey.py](https://github.com/BertonYang18/BARN-monkey/blob/main/tools/train_monkey.py) / [test_monkey.py](https://github.com/BertonYang18/BARN-monkey/blob/main/tools/test_monkey.py) / [detect_monkey.py](https://github.com/BertonYang18/BARN-monkey/blob/main/tools/detect_monkey.py).

## Original readme.md of mmaction2
The followings are the content in readme.md of mmaction2:

[📘Documentation](https://mmaction2.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmaction2.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmaction2.readthedocs.io/en/latest/modelzoo.html) |
[🆕Update News](https://mmaction2.readthedocs.io/en/latest/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmaction2/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmaction2/issues/new/choose)

# 1.1 Introduction

English | [简体中文](/README_zh-CN.md)

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)


# 1.2 Installation

Please refer to [install.md](docs/install.md) for installation.


# 1.3 Get Started

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

## Note

1- Please use the files [mmcv](https://github.com/BertonYang18/BARN-monkey/tree/main/mmcv) and [mmdet](https://github.com/BertonYang18/BARN-monkey/tree/main/mmdet) in BARN instead of the original version.

2- Please download [ext.cpython-38-x86_64-linux-gnu.so](https://drive.google.com/file/d/1fgIpmBsJdoVywg6Ax_8QenBmSFdi2nA0/view?usp=sharing) and save it to the path of "BARN-monkey/mmcv/_ext.cpython-38-x86_64-linux-gnu.so".

## Monkey detector of BARN
The [monkey detector](https://github.com/BertonYang18/BARN_detector_YOLOv7) can generate the proposals including identities and bounding boxes of socially housed macaques.Specifically, using the monkey detector as follows:
1. Using the monkey detector to generate the proposals, which is the format of txt file.
2. Using the [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) to convert the txt results into pickle (.pkl) file.
3. Adopting the target config file and GPU in train_monkey.py / test_monkey.py 
4. Checking the path of .pkl file in the config file such as [BARN.py](https://github.com/BertonYang18/BARN-monkey/blob/main/configs/detection/monkey_interaction/mix_r50_4x16x1_20e_ava_rgb_custom.py).

## Training and testing

After the above preparation, performing the following steps:
<div align="left">
  <p style="font-size:1.5vw;">1- Running the train_monkey.py / test_monkey.py to generate the behavior predictions file (.csv).</p>
  <p style="font-size:1.5vw;">2- Using the [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) to fuse the result of the monkey detector and the result of BARN into final behavior predictions (.csv file). One line of the csv file represents one of the prediction results of a bbox. (One bbox may corresponde to multiple simultaneous behaviors, as the AVA dataset does)</p>
  <p style="font-size:1.5vw;">Each line: 'videoname, timstamp, bbox, behavior, identity, confidence of behavior'</p>
  <p style="font-size:1.5vw;">3.1- Using the final behavior predictions for behavior analysis. The [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) provides the codes to generate the movement distance, duration of behaviors.</p>
  <p style="font-size:1.5vw;">3.2- The [data process code](https://github.com/BertonYang18/BARN-monkey/blob/main/data/monkey_data_process.py) provides the codes to visualize the results on imgs and concatnate them into visual videos, which could show the identity, bbox, behaviors and movement trajectory. (The code would convert the order number of behaviors in original annotations into the order number of behaviors in the paper) (There may be some bugs in the visualization process and we will correct them as soon as possible)</p>
  </div>


Although the prediction process is automatical, it may be difficult for some researchers. Thus we may develop a Graphical User Interface (GUI) like the [DEEPLUBCUT](https://github.com/BertonYang18/DeepLabCut) for application in the future.

