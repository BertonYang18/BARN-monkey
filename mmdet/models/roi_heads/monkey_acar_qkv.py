# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import build_shared_head
from .monkey_switch_backbone import monkey_switch_ResNet3dSlowFast
from .monkey_switch_head import monkey_switch_Head


class monkey_ACAR_QKV(nn.Module):
    def __init__(self, status_dim=2, input_dim=2304, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(monkey_ACAR_QKV, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  #保持dim不变
        self.conv_reduce = nn.Conv2d(input_dim, hidden_dim, 1, bias=False)
        self.conv_reduce_size = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, bias=False)
        self.relu = nn.ReLU()
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        #q
        self.status_dim = status_dim
        #self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_q1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        #k,v
        self.conv_k1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        # virt_feat
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)

        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)  # 把hidden_dim个channel分为1个group来norm
        self.dp = nn.Dropout(0.2)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(2*hidden_dim, hidden_dim, bias=False)

    def forward(self, rois, interact_feat, bbox_feat):
        # rois:(124,5)  interact_feats: (124,2048,1,16,16)   bbox_feats: (124,1024,1,8,8)  
        interact_feat = interact_feat.squeeze(2)  # -> (124,2304,16,16)
        interact_feat = self.conv_reduce(interact_feat)  # -> (124,512,16,16)
        interact_feat = self.relu(interact_feat)  # -> (124,512,16,16)
        interact_feat = self.conv_reduce_size(interact_feat)  # -> (124,512,14,14)
        interact_feat = self.downsample(interact_feat)  # -> (124,512,7,7)

        B = int(rois[-1, 0] + 1)
        r = list(rois[:, 0:1])
        roi_ids = [0]
        for i in range(B):
            n_roi = r.count(i)
            roi_id = roi_ids[i] + n_roi
            roi_ids.append(roi_id)
        assert len(roi_ids) == B + 1

        #status:(124,2)
        N, _, H, W = interact_feat.size()  # C == hidden_dim
        high_order_feats = []
        for idx in range(B):
            n_rois = roi_ids[idx+1] - roi_ids[idx]  #当前帧的roi数量
            if n_rois == 0:
                continue
            first_order_feats = interact_feat[roi_ids[idx]:roi_ids[idx+1]]

            query1 = self.conv_q1(first_order_feats).unsqueeze(1)  # ->(6,512,8,8) ->(6,1,512,8,8)
            #k1, v1   
            key1 = self.conv_k1(first_order_feats).unsqueeze(0)  # -> (1,6,512,8,8)
            att1 = (query1 * key1).sum(2) / (self.hidden_dim ** 0.5) # (6,6,8,8)
            att1 = nn.Softmax(dim=1)(att1)
            value1 = self.conv_v1(first_order_feats)  # (6,512,8,8)
            virt_feats1 = (att1.unsqueeze(2) * value1).sum(1)   # (6,512,8,8)
          
            #virt_feats1
            virt_feats1 = self.norm(virt_feats1)
            virt_feats1 = nn.functional.relu(virt_feats1)
            virt_feats1 = self.conv1(virt_feats1)
            virt_feats1 = self.dp(virt_feats1)
           
            #bbox_feat1
            high_order_feat = first_order_feats + virt_feats1 # (6,512,8,8) +=  (6,512,8,8)
            high_order_feat = self.gap1(high_order_feat) # -> (6,512,1,1)
            high_order_feat = high_order_feat.view(n_rois, -1)  # (6,512)

            high_order_feats.append(high_order_feat)
           
        high_order_feats = torch.cat(high_order_feats, dim=0)  # ->(N,512)
        out_feats = self.gap2(bbox_feat).view(N, -1)  # (N,1024,1,8,8)  -> (N,1024)
        out_feats = self.fc1(out_feats)  # -> (N,512)
        out_feats = self.relu(out_feats)
        # (N, 1024)
        out_feats = torch.cat([high_order_feats, out_feats], dim=1)
        
        return out_feats

