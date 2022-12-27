#!/usr/bin/env bash


# python build_rawframes.py ../../data/ava/videos_15min/ ../../data/ava/rawframes/ --task rgb --level 1 --mixed-ext
# python /home/yangsen/mycode/mmaction2-master/tools/data/build_rawframes.py /data/ava/videos_15min/2qQs3Y9OJX0.mkv /data/ys/ava/frames/ --task rgb --level 1 --mixed-ext
python /home/yangsen/mycode/mmaction2-master/tools/data/build_rawframes.py /data/ys/ava/aa/ /data/ys/ava/bb/ --task rgb --level 1 --mixed-ext
echo "Genearte raw frames (RGB only)"


