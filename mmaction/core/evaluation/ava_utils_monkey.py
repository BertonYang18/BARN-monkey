# Copyright (c) OpenMMLab. All rights reserved.
# This piece of code is directly adapted from ActivityNet official repo
# https://github.com/activitynet/ActivityNet/blob/master/
# Evaluation/get_ava_performance.py. Some unused codes are removed.
import csv
import logging
import time
from collections import defaultdict

import numpy as np

from .ava_evaluation_monkey import object_detection_evaluation as det_eval
from .ava_evaluation_monkey import standard_fields

import os
import json
import time
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  #16*16
        self.num_classes = num_classes
        self.labels = labels

    def update(self, new_score, gt_labels):
        assert len(new_score)==len(gt_labels)
        for k, v in new_score.items():
            #每一帧  'b5_up_2_0155_0226,0.333333'
            pd_label = v # [[1,7], [], [1], [2]]
            gt_label = gt_labels[k]  # [[1], [7], [1], [2]]
            assert len(pd_label) == len(gt_label)
            #bbox级别
            for p1, g1 in zip(pd_label, gt_label):
                num_pd = len(p1)  #[1,7] -> 2
                g1 = g1 * num_pd  #[1] * 2 = [1, 1]
                #标签级别
                for p1_a, g1_a in zip(p1, g1):
                    # p1_a, g1_a  = 1, 1;   7, 1
                    self.matrix[p1_a-1, g1_a-1] += 1


    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self, result_file):
        matrix = self.matrix
        print(matrix)

        outpath = result_file.replace('result', 'result_matrix')
        np.savetxt(outpath, matrix, delimiter=',')
        # plt.imshow(matrix, cmap=plt.cm.Blues)
        
        # # 设置x轴坐标label
        # plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # # 设置y轴坐标label
        # plt.yticks(range(self.num_classes), self.labels)
        # # 显示colorbar
        # plt.colorbar()
        # plt.xlabel('True Labels')
        # plt.ylabel('Predicted Labels')
        # plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        # thresh = matrix.max() / 2
        # for x in range(self.num_classes):
        #     for y in range(self.num_classes):
        #         # 注意这里的matrix[y, x]不是matrix[x, y]
        #         info = int(matrix[y, x])
        #         plt.text(x, y, info,
        #                  verticalalignment='center',
        #                  horizontalalignment='center',
        #                  color="white" if info > thresh else "black")
        # plt.tight_layout()
        # plt.show()

def det2csv(dataset, results, custom_classes):
    csv_results = []
    for idx in range(len(dataset)):
        video_id = dataset.video_infos[idx]['video_id']
        timestamp = dataset.video_infos[idx]['timestamp']
        result = results[idx]
        for label, _ in enumerate(result):
            for bbox in result[label]:
                bbox_ = tuple(bbox.tolist())
                if custom_classes is not None:
                    actual_label = custom_classes[label + 1]
                else:
                    actual_label = label + 1
                csv_results.append((
                    video_id,
                    timestamp,
                ) + bbox_[:4] + (actual_label, ) + bbox_[4:])
    return csv_results


# results is organized by class
def results2csv(dataset, results, out_file, custom_classes=None):
    if isinstance(results[0], list):
        csv_results = det2csv(dataset, results, custom_classes)

    # save space for float
    def to_str(item):
        if isinstance(item, float):
            return f'{item:.6f}'
        return str(item)

    with open(out_file, 'w') as f:
        for csv_result in csv_results:
            f.write(','.join(map(to_str, csv_result)))
            f.write('\n')


def print_time(message, start):
    print('==> %g seconds to %s' % (time.time() - start, message), flush=True)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return f'{video_id},{float(timestamp):.6f}'


def read_csv(csv_file, class_whitelist=None):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
        csv_file: A file object.
        class_whitelist: If provided, boxes corresponding to (integer) class
        labels not in this set are skipped.

    Returns:
        boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
        labels: A dictionary mapping each unique image key (string) to a list
        of integer class labels, matching the corresponding box in `boxes`.
        scores: A dictionary mapping each unique image key (string) to a list
        of score values labels, matching the corresponding label in `labels`.
        If scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)  #csv_fil可能为 标注的train/val.csv，也可能为预测的predict.csv
    for row in reader:
        assert len(row) in [7, 8, 9], 'Wrong number of columns: ' + row  
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue
        #predict.csv： videoname, time, bbox*4, action_id, action_score
        #monkey_val中为 videoname, time, bbox*4, action_id, person_id, max_frame_num; 没有score
        #当打开的时anno文件时是不可能有score的； predict.csv会有
        score = 1.0
        if len(row) == 8:
            score = float(row[7])

        entries[image_key].append((score, action_id, y1, x1, y2, x2))

    for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: (tup[2], tup[3], tup[4], tup[5], tup[1],-tup[0]))
        boxes[image_key] = [x[2:] for x in entry]
        labels[image_key] = [x[1] for x in entry]
        scores[image_key] = [x[0] for x in entry]
        if image_key ==  'b5_up_2_0155_0226,0.333333':
            print()

    print_time('read file ' + csv_file.name, start)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
        exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
        A set of strings containing excluded image keys, e.g.
        "aaaaaaaaaaa,0904",
        or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
    for row in reader:
        assert len(row) == 2, f'Expected only 2 columns, got: {row}'
        excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
        labelmap_file: A file object containing a label map protocol buffer.

    Returns:
        labelmap: The label map in the form used by the
        object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
        class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids


# Seems there is at most 100 detections for each image
def ava_eval(result_file,
             result_type,
             label_file,
             ann_file,
             exclude_file,
             verbose=True,
             custom_classes=None):

    assert result_type in ['mAP']

    start = time.time()
    categories, class_whitelist = read_labelmap(open(label_file))
    if custom_classes is not None:
        custom_classes = custom_classes[1:]
        assert set(custom_classes).issubset(set(class_whitelist))
        class_whitelist = custom_classes
        categories = [cat for cat in categories if cat['id'] in custom_classes]

    # loading gt, do not need gt score
    gt_boxes, gt_labels, _ = read_csv(open(ann_file), class_whitelist)
    if verbose:
        print_time('Reading detection results', start)

    if exclude_file is not None:
        excluded_keys = read_exclusions(open(exclude_file))
    else:
        excluded_keys = list()

    start = time.time()
    boxes, labels, scores = read_csv(open(result_file), class_whitelist)
    if verbose:
        print_time('Reading detection results', start)

    # Evaluation for mAP
    pascal_evaluator = det_eval.PascalDetectionEvaluator(categories)

    start = time.time()
    for image_key in gt_boxes:
        if verbose and image_key in excluded_keys:
            logging.info(
                'Found excluded timestamp in detections: %s.'
                'It will be ignored.', image_key)
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key, {
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array(gt_boxes[image_key], dtype=float),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array(gt_labels[image_key], dtype=int)
            })
    if verbose:
        print_time('Convert groundtruth', start)

    start = time.time()
    for image_key in boxes:
        if verbose and image_key in excluded_keys:
            logging.info(
                'Found excluded timestamp in detections: %s.'
                'It will be ignored.', image_key)
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key, {
                standard_fields.DetectionResultFields.detection_boxes:
                np.array(boxes[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes:
                np.array(labels[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores:
                np.array(scores[image_key], dtype=float)
            })
    if verbose:
        print_time('convert detections', start)

 
    start = time.time()
    metrics = pascal_evaluator.evaluate()
    if verbose:
        print_time('run_evaluator', start)
    for display_name in metrics:
        print(f'{display_name}=\t{metrics[display_name]}')
    for display_name in metrics:
        logging.info((f'{display_name}=\t{metrics[display_name]}'))
    return {
        display_name: metrics[display_name]
        for display_name in metrics if 'ByCategory' not in display_name
    }

