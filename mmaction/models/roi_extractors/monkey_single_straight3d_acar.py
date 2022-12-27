# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.utils import import_module_error_class
import torchvision
try:
    from mmcv.ops import RoIAlign, RoIPool
except (ImportError, ModuleNotFoundError):

    @import_module_error_class('mmcv-full')
    class RoIAlign(nn.Module):
        pass

    @import_module_error_class('mmcv-full')
    class RoIPool(nn.Module):
        pass


try:
    from mmdet.models import ROI_EXTRACTORS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class monkey_SingleRoIExtractor3D_acar(nn.Module):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type. Default: 'RoIAlign'.
        featmap_stride (int): Strides of input feature maps. Default: 16.
        output_size (int | tuple): Size or (Height, Width). Default: 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Default: 0.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
            Default: 'avg'.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Default: True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Default: False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.
    """

    def __init__(self,
                 roi_layer_type='RoIAlign',
                 featmap_stride=16,
                 output_size=16,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 with_temporal_pool=True,
                 temporal_pool_mode='avg',
                 with_global=False):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        # if self.roi_layer_type == 'RoIPool':
        #     self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        # else:
        #     self.roi_layer = RoIAlign(
        #         self.output_size,
        #         self.spatial_scale,
        #         sampling_ratio=self.sampling_ratio,
        #         pool_mode=self.pool_mode,
        #         aligned=self.aligned)
        # self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def init_weights(self):
        pass

    # The shape of feat is N, C, T, H, W
    def forward(self, feat, rois):
        if not isinstance(feat, tuple):
            feat = (feat, )

        roi_features = []
        for f in feat:
            sp = f.shape
            h, w = sp[3:]
            # f: [2,2048,8,16,29]  
            # nn.AdaptiveAvgPool3d((1, h, w))(f):  -> [2,2048,1,16,29]
            # .view(-1, sp[1], h, w)  -> [2, 2048, 16, 29]
            feats = nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, sp[1], h, w)

            rois = rois.clone()  #创建副本，指向新的内存地址。
            rois[:, 1] = rois[:, 1] * w
            rois[:, 2] = rois[:, 2] * h
            rois[:, 3] = rois[:, 3] * w
            rois[:, 4] = rois[:, 4] * h
            rois = rois.detach()  #新的tensor和原来的tensor共享数据内存，但不涉及梯度计算
            # feats:[2, 2048, 16, 29]   rois: [6,5]   self.roi_spatial=7
            # torchvision.ops.roi_align ->  [6, 2048, 7, 7]
            # self.roi_maxpool -> [6, 2048, 1]
            # view -> [6,2048]
            roi_feats = torchvision.ops.roi_align(feats, rois, (7, 7))
            # roi_feats = self.roi_maxpool(roi_feats).view(-1, sp[1]) #*****

            roi_features.append(roi_feats)
        # roi_features:[[6,2048, 7, 7], [6,256], 7, 7]
        # torch.cat -> [6, 2304, 7, 7]
        roi_features = torch.cat(roi_features, dim=1)

        return roi_features, feat
        # return torch.stack(roi_features, dim=2), feat  # 增加一个新的维度来拼接： [(3,3), [3,3], dim=0) -> (2,3,3)


if mmdet_imported:
    ROI_EXTRACTORS.register_module()(monkey_SingleRoIExtractor3D_acar)
