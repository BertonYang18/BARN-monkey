# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Functions for computing metrics like precision, recall, CorLoc and etc."""

import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

def compute_precision_recall(scores, labels, num_gt):
    """Compute precision and recall.

    Args:
        scores: A float numpy array representing detection score
        labels: A boolean numpy array representing true/false positive labels
        num_gt: Number of ground truth instances

    Raises:
        ValueError: if the input is not of the correct format

    Returns:
        precision: Fraction of positive instances over detected ones. This
            value is None if no ground truth labels are present.
        recall: Fraction of detected positive instance over all positive
            instances. This value is None if no ground truth labels are
            present.
    """
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.bool
            or len(labels.shape) != 1):
        raise ValueError('labels must be single dimension bool numpy array')

    if not isinstance(scores, np.ndarray) or len(scores.shape) != 1:
        raise ValueError('scores must be single dimension numpy array')

    if num_gt < np.sum(labels):
        raise ValueError(
            'Number of true positives must be smaller than num_gt.')

    if len(scores) != len(labels):
        raise ValueError('scores and labels must be of the same size.')

    if num_gt == 0:
        return None, None

    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)
    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)
    precision = cum_true_positives.astype(float) / (
        cum_true_positives + cum_false_positives)
    recall = cum_true_positives.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    """Compute Average Precision according to the definition in VOCdevkit.

    Precision is modified to ensure that it does not decrease as recall
    decrease.

    Args:
        precision: A float [N, 1] numpy array of precisions
        recall: A float [N, 1] numpy array of recalls

    Raises:
        ValueError: if the input is not of the correct format

    Returns:
        average_precison: The area under the precision recall curve. NaN if
            precision and recall are None.
    """
    if precision is None:
        if recall is not None:
            raise ValueError('If precision is None, recall must also be None')
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(
            recall, np.ndarray):
        raise ValueError('precision and recall must be numpy array')
    if precision.dtype != np.float or recall.dtype != np.float:
        raise ValueError('input must be float numpy array.')
    if len(precision) != len(recall):
        raise ValueError('precision and recall must be of the same size.')
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError('Precision must be in the range of [0, 1].')
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError('recall must be in the range of [0, 1].')
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError('recall must be a non-decreasing array')

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
    """Compute CorLoc according to the definition in the following paper.

    https://www.robots.ox.ac.uk/~vgg/rg/papers/deselaers-eccv10.pdf

    Returns nans if there are no ground truth images for a class.

    Args:
        num_gt_imgs_per_class: 1D array, representing number of images
            containing at least one object instance of a particular class
        num_images_correctly_detected_per_class: 1D array, representing number
            of images that are correctly detected at least one object instance
            of a particular class

    Returns:
        corloc_per_class: A float numpy array represents the corloc score of
            each class
    """
    # Divide by zero expected for classes with no gt examples.
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            num_gt_imgs_per_class == 0, np.nan,
            num_images_correctly_detected_per_class / num_gt_imgs_per_class)


def compute_fpr_fnr(scores, labels,
                num_Truth_per_class,
                num_False_per_class,
                top_k=(1,3,5),
                thr=0.5):
    """Compute precision and recall.

    Args:
        scores: A float numpy array representing detection score
        labels: A boolean numpy array representing true/false positive labels
        num_gt: Number of ground truth instances

    Raises:
        ValueError: if the input is not of the correct format

    Returns:
        precision: Fraction of positive instances over detected ones. This
            value is None if no ground truth labels are present.
        recall: Fraction of detected positive instance over all positive
            instances. This value is None if no ground truth labels are
            present.
    """
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.bool
            or len(labels.shape) != 1):
        raise ValueError('labels must be single dimension bool numpy array')

    if not isinstance(scores, np.ndarray) or len(scores.shape) != 1:
        raise ValueError('scores must be single dimension numpy array')

    if num_Truth_per_class < np.sum(labels):
        raise ValueError(
            'Number of true positives must be smaller than num_gt.')

    if len(scores) != len(labels):
        raise ValueError('scores and labels must be of the same size.')

    if num_Truth_per_class == 0:
        return None, None

    

    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)

    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    # TP FP
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)

    cum_positive = cum_true_positives + cum_false_positives
    precision = cum_true_positives.astype(float) / (cum_positive)
    recall = cum_true_positives.astype(float) / num_Truth_per_class

    num_False_per_class = len(labels) - num_Truth_per_class 
    cum_false_negative = num_Truth_per_class - cum_true_positives
    cum_true_negative = num_False_per_class - cum_false_positives

    fpr = cum_false_positives.astype(float) / num_False_per_class
    fnr = cum_false_negative.astype(float) / num_Truth_per_class

    

    '''thr mehotd1'''
    thr_labels = labels[sorted_indices]
    thr_scores = scores[sorted_indices]  # now, thr_scores and labels are both reversed
    thr_scores = np.array([i for i in thr_scores if i > thr])
    num_higher_thr = len(thr_scores)
    thr_precision, thr_recall = precision[num_higher_thr-1], recall[num_higher_thr-1]
    thr_fpr, thr_fnr = fpr[num_higher_thr-1], fnr[num_higher_thr-1]
    '''thr mehotd2'''
    # fpr_standard, tpr_standard, thresholds = roc_curve(labels, scores, pos_label=1,  drop_intermediate=False)
    # for i in range(thresholds.shape[0]):
    #     if thresholds[i] <thr:
    #         break
    # thr_fpr2, thr_fnr2 = fpr_standard[i-1], tpr_standard[i-1]

    '''top_k'''
    k_precisions = []
    k_recalls = []
    k_fprs = []
    k_fnrs = []
    for k in top_k:
        k_precision, k_recall = precision[k-1], recall[k-1]
        k_fpr, k_fnr = fpr[k-1], fnr[k-1]

        k_precisions.append(k_precision)
        k_recalls.append(k_recall)
        k_fprs.append(k_fpr)
        k_fnrs.append(k_fnr)

    return (precision, recall, fpr, fnr), (thr_precision, thr_recall, thr_fpr, thr_fnr), (k_precisions, k_recalls, k_fprs, k_fnrs)


# def compute_auc(fpr, tpr):
#     """Compute Average Precision according to the definition in VOCdevkit.

#     Precision is modified to ensure that it does not decrease as recall
#     decrease.

#     Args:
#         precision: A float [N, 1] numpy array of precisions
#         recall: A float [N, 1] numpy array of recalls

#     Raises:
#         ValueError: if the input is not of the correct format

#     Returns:
#         average_precison: The area under the precision recall curve. NaN if
#             precision and recall are None.
#     """
#     if precision is None:
#         if recall is not None:
#             raise ValueError('If precision is None, recall must also be None')
#         return np.NAN

#     if not isinstance(precision, np.ndarray) or not isinstance(
#             recall, np.ndarray):
#         raise ValueError('precision and recall must be numpy array')
#     if precision.dtype != np.float or recall.dtype != np.float:
#         raise ValueError('input must be float numpy array.')
#     if len(precision) != len(recall):
#         raise ValueError('precision and recall must be of the same size.')
#     if not precision.size:
#         return 0.0
#     if np.amin(precision) < 0 or np.amax(precision) > 1:
#         raise ValueError('Precision must be in the range of [0, 1].')
#     if np.amin(recall) < 0 or np.amax(recall) > 1:
#         raise ValueError('recall must be in the range of [0, 1].')
#     if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
#         raise ValueError('recall must be a non-decreasing array')

#     recall = np.concatenate([[0], recall, [1]])
#     precision = np.concatenate([[0], precision, [0]])

#     # Preprocess precision to be a non-decreasing array
#     for i in range(len(precision) - 2, -1, -1):
#         precision[i] = np.maximum(precision[i], precision[i + 1])

#     indices = np.where(recall[1:] != recall[:-1])[0] + 1
#     average_precision = np.sum(
#         (recall[indices] - recall[indices - 1]) * precision[indices])
#     return average_precision



def compute_roc_auc_score(scores, labels, average='micro'):
    
    return roc_auc_score(labels,scores, average='micro')

def compute_AUC(roc_all_class):
    num_class = len(roc_all_class)
    roc = 0
    for i in range(num_class):
        roc += float(roc_all_class[i])
    return roc / (num_class + 1)