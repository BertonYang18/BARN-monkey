# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import build_shared_head
from .monkey_switch_backbone import monkey_switch_ResNet3dSlowFast
from .monkey_switch_head import monkey_switch_Head


class monkey_switch_group_newQKV_status5(nn.Module):
    def __init__(self, status_dim=2, input_dim=2304, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(monkey_switch_group_newQKV_status5, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  #保持dim不变
        self.conv_reduce = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        #q
        self.status_dim = status_dim
        #self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_q1 = nn.Conv2d(status_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_q2 = nn.Conv2d(status_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        #k,v
        self.conv_k1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        # virt_feat
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)

        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)  # 把hidden_dim个channel分为1个group来norm
        self.dp = nn.Dropout(0.2)

    def forward(self, status, rois, bbox_feat):
        #bbox_feat:(124,2304,1,8,8)
        bbox_feat = bbox_feat.squeeze(2)  # -> (124,2304,8,8)
        bbox_feat = self.conv_reduce(bbox_feat)  # -> (124,512,8,8)

        B = int(rois[-1, 0] + 1)
        r = list(rois[:, 0:1])
        roi_ids = [0]
        for i in range(B):
            n_roi = r.count(i)
            roi_id = roi_ids[i] + n_roi
            roi_ids.append(roi_id)
        assert len(roi_ids) == B + 1

        #status:(124,2)
        N, _, H, W = bbox_feat.size()  # C == hidden_dim
        group_bbox_feats1 = []
        group_bbox_feats2 = []
        for idx in range(B):
            n_rois = roi_ids[idx+1] - roi_ids[idx]  #当前帧的roi数量
            if n_rois == 0:
                continue
            actor_feats = bbox_feat[roi_ids[idx]:roi_ids[idx+1]]
            actor_bboxs = rois[roi_ids[idx]:roi_ids[idx+1], 1:]
            actor_guide = status[roi_ids[idx]:roi_ids[idx+1]]

            status1 = torch.cat((actor_guide[:,0].reshape(n_rois, 1), actor_bboxs), dim=1)  #(124,2) + (124,4) -> (124, 6)
            status1 = status1.unsqueeze(2).unsqueeze(2).expand(n_rois,self.status_dim,H,W)  # ->(124,6,8,8)
            status2 = torch.cat((actor_guide[:,1].reshape(n_rois, 1), actor_bboxs), dim=1)  #(124,2) + (124,4) -> (124, 6)
            status2 = status2.unsqueeze(2).unsqueeze(2).expand(n_rois,self.status_dim,H,W)  # ->(124,6,8,8)

            # status1 = actor_bboxs  #(124,2) + (124,4) -> (124, 6)
            # status1 = status1.unsqueeze(2).unsqueeze(2).expand(n_rois,self.status_dim,H,W)  # ->(124,6,8,8)
            # status2 = actor_bboxs  #(124,2) + (124,4) -> (124, 6)
            # status2 = status2.unsqueeze(2).unsqueeze(2).expand(n_rois,self.status_dim,H,W)  # ->(124,6,8,8)
            #q1 q2
            #query = self.conv_q(status).unsqueeze(1)
            query1 = self.conv_q1(status1).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
            query2 = self.conv_q2(status2).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
            #k1, v1   
            key1 = self.conv_k1(actor_feats).unsqueeze(0)  # -> (1，124,512,8,8)
            att1 = (query1 * key1).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
            att1 = nn.Softmax(dim=1)(att1)
            value1 = self.conv_v1(actor_feats)  # (124,512,8,8)
            virt_feats1 = (att1.unsqueeze(2) * value1).sum(1)   # (124,512,8,8)
            #k2, v2
            key2 = self.conv_k2(actor_feats).unsqueeze(0)  # -> (1，124,512,8,8)
            att2 = (query2 * key2).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
            att2 = nn.Softmax(dim=1)(att2)
            value2 = self.conv_v2(actor_feats)  # (124,512,8,8)
            virt_feats2 = (att2.unsqueeze(2) * value2).sum(1)   # (124,512,8,8)
            #virt_feats1
            virt_feats1 = self.norm(virt_feats1)
            virt_feats1 = nn.functional.relu(virt_feats1)
            virt_feats1 = self.conv1(virt_feats1)
            virt_feats1 = self.dp(virt_feats1)
            #virt_feats2
            virt_feats2 = self.norm(virt_feats2)
            virt_feats2 = nn.functional.relu(virt_feats2)
            virt_feats2 = self.conv2(virt_feats2)
            virt_feats2 = self.dp(virt_feats2)
            #bbox_feat1
            group_bbox_feat1 = actor_feats + virt_feats1 # (124,512,8,8) +=  (124,512,8,8)
            group_bbox_feat1 = group_bbox_feat1.unsqueeze(2)  # (124,512,1,8,8)
            group_bbox_feats1.append(group_bbox_feat1)
            #bbox_feat2
            group_bbox_feat2 = actor_feats + virt_feats2 # (124,512,8,8) +=  (124,512,8,8)
            group_bbox_feat2 = group_bbox_feat2.unsqueeze(2)  # (124,512,1,8,8)
            group_bbox_feats2.append(group_bbox_feat2)

        group_bbox_feats1 = torch.cat(group_bbox_feats1, dim=0)  # ->(N,512,1,8,8)
        group_bbox_feats2 = torch.cat(group_bbox_feats2, dim=0)  # ->(N,512,1,8,8)
        gropu_bbox_feats = [group_bbox_feats1, group_bbox_feats2]
        return gropu_bbox_feats

