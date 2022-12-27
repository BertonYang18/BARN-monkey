#!/usr/bin/env bash

cd ../
#python build_rawframes.py ../../data/ava/videos_15min/ ../../data/ava/rawframes/ --task both --level 1 --flow-type tvl1 --mixed-ext
python /home/yangsen/mycode/mmaction2-master/tools/data/build_rawframes.py /data/ys/monkey/alldata/videos/ /data/ys/monkey/alldata/rawframes/ --task both --level 1 --flow-type tvl1 --mixed-ext

echo "Raw frames (RGB and Flow) Generated"
cd ava/
