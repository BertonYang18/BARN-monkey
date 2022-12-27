# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.core.bbox import bbox_target

# try:
#     from mmdet.models.builder import HEADS as MMDET_HEADS
#     mmdet_imported = True
# except (ImportError, ModuleNotFoundError):
#     mmdet_imported = False

# Resolve cross-entropy function to support multi-target in Torch < 1.10
#   This is a very basic 'hack', with minimal functionality to support the
#   procedure under prior torch versions
from packaging import version as pv

if pv.parse(torch.__version__) < pv.parse('1.10'):

    def cross_entropy_loss(input, target, reduction='None'):
        input = input.log_softmax(dim=-1)  # Compute Log of Softmax
        loss = -(input * target).sum(dim=-1)  # Compute Loss manually
        if reduction.lower() == 'mean':
            return loss.mean()
        elif reduction.lower() == 'sum':
            return loss.sum()
        else:
            return loss
else:
    cross_entropy_loss = F.cross_entropy


class monkey_switch_Head(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively.

    Args:
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
        in_channels (int): The number of input channels. Default: 2048.
        focal_alpha (float): The hyper-parameter alpha for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 1.
        focal_gamma (float): The hyper-parameter gamma for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 0.
        num_classes (int): The number of classes. Default: 81.
        dropout_ratio (float): A float in [0, 1], indicates the dropout_ratio.
            Default: 0.
        dropout_before_pool (bool): Dropout Feature before spatial temporal
            pooling. Default: True.
        topk (int or tuple[int]): Parameter for evaluating Top-K accuracy.
            Default: (3, 5)
        multilabel (bool): Whether used for a multilabel task. Default: True.
    """

    def __init__(
            self,
            temporal_pool_type='avg',
            spatial_pool_type='max',
            in_channels=2048,
            focal_gamma=0.,
            focal_alpha=1.,
            num_classes=81,  # First class reserved (BBox as pos/neg)
            dropout_ratio=0,
            dropout_before_pool=True,
            topk=(1,),
            multilabel=True):

        super(monkey_switch_Head, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk, )
        elif isinstance(topk, tuple):
            assert all([isinstance(k, int) for k in topk])
            self.topk = topk
        else:
            raise TypeError('topk should be int or tuple[int], '
                            f'but get {type(topk)}')
        # Class 0 is ignored when calculating accuracy,
        #      so topk cannot be equal to num_classes.
        assert all([k < num_classes for k in self.topk])

        in_channels = self.in_channels
        # Pool by default
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.debug_imgs = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = self.temporal_pool(x)
        x = self.spatial_pool(x)

        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        status = self.fc_cls(x)
        # We do not predict bbox, so return None
        # return cls_score, None
        return status

   