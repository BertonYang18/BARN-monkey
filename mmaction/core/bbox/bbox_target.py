# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def bbox_target(pos_bboxes_list, neg_bboxes_list, gt_labels, cfg):
    """Generate classification targets for bboxes.

    Args:
        pos_bboxes_list (list[Tensor]): Positive bboxes list.
        neg_bboxes_list (list[Tensor]): Negative bboxes list.
        gt_labels (list[Tensor]): Groundtruth classification label list.
        cfg (Config): RCNN config.

    Returns:
        (Tensor, Tensor): Label and label_weight for bboxes.
    """
    labels, label_weights = [], []
    pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight

    assert len(pos_bboxes_list) == len(neg_bboxes_list) == len(gt_labels)
    length = len(pos_bboxes_list)  # batch_size 

    for i in range(length):
        pos_bboxes = pos_bboxes_list[i]
        neg_bboxes = neg_bboxes_list[i]
        gt_label = gt_labels[i]

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        label = F.pad(gt_label, (0, 0, 0, num_neg))  # gt_label即pos_gt_labels：(4，17)  len(gt_label)== 4 ==num_pos; 假如num_neg=2, 则label:(6,17),默认value=0，用0填充
        label_weight = pos_bboxes.new_zeros(num_samples)  # pos_bboxes == torch.tensor(())   可以被空的tensor替换
        label_weight[:num_pos] = pos_weight
        label_weight[-num_neg:] = 1.  # len(label_weight) == num_pos + num_neg

        labels.append(label)
        label_weights.append(label_weight)  # list[tensor]   len(list)==batch  tensor.shape(num_pos+num_neg)

    labels = torch.cat(labels, 0)  # tensor.shape(65,17)
    label_weights = torch.cat(label_weights, 0)  # tensor.shape(65)   每个bbox对应一个权重，权重为 1.
    return labels, label_weights
