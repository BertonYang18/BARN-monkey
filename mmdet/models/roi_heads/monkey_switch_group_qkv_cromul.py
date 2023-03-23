# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import build_shared_head
from .monkey_switch_backbone import monkey_switch_ResNet3dSlowFast
from .monkey_switch_head import monkey_switch_Head


class monkey_switch_group_QKV_cromul(nn.Module):
    def __init__(self, status_dim=2, input_dim=2304, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(monkey_switch_group_QKV_cromul, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  #保持dim不变
        self.conv_reduce = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        #q
        self.status_dim = status_dim
        self.q1 = nn.Linear(status_dim, hidden_dim, bias=False)
        self.q2 = nn.Linear(status_dim, hidden_dim, bias=False)
        # #self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # self.conv_q1 = nn.Conv2d(status_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # self.conv_q2 = nn.Conv2d(status_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # #k,v
        # self.conv_k1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # self.conv_v1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # self.conv_k2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # self.conv_v2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.k1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # virt_feat
        # self.conv = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)  # 把hidden_dim个channel分为1个group来norm  input(N,C,*)
        self.dp = nn.Dropout(0.2)

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, status, rois, bbox_feat):
        #bbox_feat:(124,2304,1,8,8)
        bbox_feat = bbox_feat.squeeze(2)  # -> (124,2304,8,8)
        bbox_feat = self.conv_reduce(bbox_feat)  # -> (124,512,8,8)
        
        #status:(124,2)
        B, C, H, W = bbox_feat.size()  # C == hidden_dim   B,C,8,8
        HW = H * W 
        status = torch.cat((status, rois[:, 1:]), dim=1)  #(124,2) + (124,4) -> (124, 6)
        status = status.unsqueeze(2).expand(B,self.status_dim,HW)  # ->(124,6,64)
        
        #q1 q2
        #query = self.conv_q(status).unsqueeze(1)
        bbox_feat_cp = bbox_feat.reshape(B, C, HW).permute(0,2,1) #B, HW, C   
        status = status.permute(0,2,1) #B, HW, C
        query1 = self.q1(status)
        query2 = self.q1(status)
        #k1, v1   
        key1 = self.k1(bbox_feat_cp)  #B, HW, C  
        keyT1 = key1.transpose(-2, -1)  #B, C, HW  
        att1 = (query1 @ keyT1) / (self.hidden_dim ** 0.5) #B, HW, HW 
        att1 = self.softmax(att1) #B, HW, HW 
        value1 = self.v1(bbox_feat_cp)  #B, HW, C  
        virt_feats1 = att1 @ value1  #B, HW, C 
        #k2, v2
        key2 = self.k2(bbox_feat_cp)  #B, HW, C  
        keyT2 = key2.transpose(-2, -1)  #B, C, HW  
        att2 = (query2 @ keyT2) / (self.hidden_dim ** 0.5) #B, HW, HW 
        att2 = self.softmax(att2) #B, HW, HW 
        value2 = self.v2(bbox_feat_cp)  #B, HW, C  
        virt_feats2 = att2 @ value2  #B, HW, C 
        #virt_feats1
        #virt_feats1 = self.norm(virt_feats1)
        #virt_feats1 = nn.functional.relu(virt_feats1)
        virt_feats1 = self.proj1(virt_feats1)
        virt_feats1 = self.dp(virt_feats1)  #B, HW, C 
        #virt_feats2
        #virt_feats2 = self.norm(virt_feats2)
        #virt_feats2 = nn.functional.relu(virt_feats2)
        virt_feats2 = self.proj2(virt_feats2)
        virt_feats2 = self.dp(virt_feats2)  #B, HW, C 
        #bbox_feat1
        group_bbox_feat1 = bbox_feat_cp + virt_feats1  #B, HW, C 
        group_bbox_feat1 = group_bbox_feat1.permute(0,2,1).reshape(B, C, H, W)
        group_bbox_feat1 = group_bbox_feat1.unsqueeze(2)  # (124,512,1,8,8)
        #bbox_feat2
        group_bbox_feat2 = bbox_feat_cp + virt_feats2 #B, HW, C 
        group_bbox_feat2 = group_bbox_feat2.permute(0,2,1).reshape(B, C, H, W)
        group_bbox_feat2 = group_bbox_feat2.unsqueeze(2)  # (124,512,1,8,8)

        gropu_bbox_feats = [group_bbox_feat1, group_bbox_feat2]
        return gropu_bbox_feats

