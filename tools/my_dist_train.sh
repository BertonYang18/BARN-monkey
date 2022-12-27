#!/usr/bin/env bash



python -m torch.distributed.launch  \
     tools/train.py  --launcher pytorch  --config configs/detection/monkey/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py
# Any arguments from the third one are captured by ${@:3}
