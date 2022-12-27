# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import build_shared_head
from .monkey_switch_backbone import monkey_switch_ResNet3dSlowFast
from .monkey_switch_head import monkey_switch_Head
from .monkey_switch_qkv import monkey_switch_QKV
from .monkey_switch_group_qkv import monkey_switch_group_QKV
from .monkey_switch_group_qkv_onlyroi import monkey_switch_group_QKV_onlyroi
from .monkey_switch_group_qkv_nostatus import monkey_switch_group_QKV_nostatus
from .monkey_switch_group_qkv_status5 import monkey_switch_group_QKV_status5
from .monkey_switch_group_newqkv_status5 import monkey_switch_group_newQKV_status5
from .monkey_switch_group_qkv_double import monkey_switch_group_QKV_double
from .monkey_switch_group_qkv_double3 import monkey_switch_group_QKV_double3
from .monkey_switch_group_qkv_three import monkey_switch_group_QKV_three
from .monkey_switch_group_qkv_double_inter import monkey_switch_group_QKV_double_inter
from .monkey_switch_group_qkv_cromul import monkey_switch_group_QKV_cromul
from .monkey_switch_group_qkv_cromul_conv import monkey_switch_group_QKV_cromul_conv

class monkey_BaseRoIHead_switch(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(monkey_BaseRoIHead_switch, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()

        #switch layers 
        #self.monkey_switch_ResNet3dSlowFast = monkey_switch_ResNet3dSlowFast(pretrained=None)
        self.switch_head = monkey_switch_Head(in_channels=2304, num_classes=2) # swBB -> status
        #self.switch_qkv = monkey_switch_QKV(status_dim=2, input_dim=2304)
        #status, rois, bbox_feat -> group_bbox_feats
        #self.switch_group_qkv = monkey_switch_group_QKV_cromul(status_dim=6, input_dim=2304) 
        #self.switch_group_qkv = monkey_switch_group_QKV_status5(status_dim=5, input_dim=2304) #status, rois, bbox_feat -> group_bbox_feats
        self.switch_group_qkv = monkey_switch_group_newQKV_status5(status_dim=5, input_dim=2304) #status, rois, bbox_feat -> group_bbox_feats
        
        pass
        
    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False,
                                **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
