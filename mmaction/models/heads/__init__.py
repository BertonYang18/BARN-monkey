# Copyright (c) OpenMMLab. All rights reserved.
from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .bbox_head import BBoxHeadAVA
from .fbo_head import FBOHead
from .i3d_head import I3DHead
from .lfb_infer_head import LFBInferHead
from .misc_head import ACRNHead
from .roi_head import AVARoIHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .stgcn_head import STGCNHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead

from .monkey_bbox_head import monkey_BBoxHeadAVA
from .monkey_bbox_head_acar import monkey_BBoxHeadAVA_acar
from .monkey_bbox_head_switch import monkey_BBoxHeadAVA_switch
from .monkey_bbox_head_switch_group import monkey_BBoxHeadAVA_switch_group
from .monkey_bbox_head_switch_group_newloss import monkey_BBoxHeadAVA_switch_group_newloss
from .monkey_bbox_head_switch_group_newloss_custom import monkey_BBoxHeadAVA_switch_group_newloss_custom
from .monkey_bbox_head_switch_group_16cat16 import monkey_BBoxHeadAVA_switch_group_16cat16

from .monkey_bbox_head_switch_group_linear import monkey_BBoxHeadAVA_switch_group_linear

from .monkey_roi_head import monkey_AVARoIHead
from .monkey_roi_head_acar_HR2O import monkey_AVARoIHead_acar_HR2O
from .monkey_roi_head_switch import monkey_AVARoIHead_switch
from .monkey_roi_head_switch_group_swBB import monkey_AVARoIHead_switch_group_swBB



__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead', 'BBoxHeadAVA', 'AVARoIHead',
    'FBOHead', 'LFBInferHead', 'TRNHead', 'TimeSformerHead', 'ACRNHead',
    'STGCNHead',

    'monkey_BBoxHeadAVA','monkey_BBoxHeadAVA_acar','monkey_BBoxHeadAVA_switch',
    'monkey_BBoxHeadAVA_switch_group','monkey_BBoxHeadAVA_switch_group_linear',
    'monkey_BBoxHeadAVA_switch_group_newloss',"monkey_BBoxHeadAVA_switch_group_16cat16",
    "monkey_BBoxHeadAVA_switch_group_newloss_custom",

    'monkey_AVARoIHead','monkey_AVARoIHead_acar_HR2O','monkey_AVARoIHead_switch',
    'monkey_AVARoIHead_switch_group_swBB'
]
