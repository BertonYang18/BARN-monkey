# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import build_shared_head
from .monkey_switch_backbone import monkey_switch_ResNet3dSlowFast
from .monkey_switch_head import monkey_switch_Head


class monkey_switch_group_QKV_cromul_conv(nn.Module):
    def __init__(self, status_dim=2, input_dim=2304, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(monkey_switch_group_QKV_cromul_conv, self).__init__()
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
        self.proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)  # 把hidden_dim个channel分为1个group来norm  input(N,C,*)
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.2)
        self.dp11 = nn.Dropout(0.2)
        self.dp22 = nn.Dropout(0.2)

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, status, rois, bbox_feat):
        #bbox_feat:(124,2304,1,8,8)
        bbox_feat = bbox_feat.squeeze(2)  # -> (124,2304,8,8)
        bbox_feat = self.conv_reduce(bbox_feat)  # -> (124,512,8,8)
        
        #status:(124,2)
        B, C, H, W = bbox_feat.size()  # C == hidden_dim   B,C,8,8
        HW = H * W 
        status1 = torch.cat((status[:,0].reshape(B,1), rois[:, 1:]), dim=1)  #(124,2) + (124,4) -> (124, 6)
        status1 = status1.unsqueeze(2).unsqueeze(2).expand(B,self.status_dim,H,W)  # ->(124,6,8,8)
        status2 = torch.cat((status[:,1].reshape(B,1), rois[:, 1:]), dim=1)  #(124,2) + (124,4) -> (124, 6)
        status2 = status2.unsqueeze(2).unsqueeze(2).expand(B,self.status_dim,H,W)  # ->(124,6,8,8)    
        #q1 q2
        #query = self.conv_q(status).unsqueeze(1)
        # bbox_feat_cp = bbox_feat.reshape(B, C, HW) #B, C, HW   
        # status = status #B, C, HW 
        query1 = self.conv_q1(status1).reshape(B,C,HW)   #B, C, HW  
        query2 = self.conv_q1(status2).reshape(B,C,HW)
        #k1, v1   
        key1 = self.conv_k1(bbox_feat).reshape(B,C,HW)  #B, C, HW  
        keyT1 = key1.transpose(-2, -1)  #B, HW, C  
        att1 = (query1 @ keyT1) / (self.hidden_dim ** 0.5) #B, C, C
        att1 = self.softmax(att1) #B, C, C
        value1 = self.conv_v1(bbox_feat).reshape(B,C,HW)   #B, C, HW  
        virt_feats1 = att1 @ value1   #B, C, HW 
        virt_feats1 = virt_feats1.reshape(B,C,H,W)
        #k2, v2
        key2 = self.conv_k2(bbox_feat).reshape(B,C,HW)  #B, C, HW  
        keyT2 = key2.transpose(-2, -1)  #B, HW, C  
        att2 = (query2 @ keyT2) / (self.hidden_dim ** 0.5) #B, C, C
        att2 = self.softmax(att2) #B, C, C
        value2 = self.conv_v2(bbox_feat).reshape(B,C,HW)   #B, C, HW  
        virt_feats2 = att2 @ value2   #B, C, HW 
        virt_feats2 = virt_feats2.reshape(B,C,H,W)
        #virt_feats1
        virt_feats1 = self.norm(virt_feats1)  #B, C, H, W  
        #virt_feats1 = nn.functional.relu(virt_feats1)
        virt_feats1 = self.conv1(virt_feats1)
        virt_feats1 = self.dp1(virt_feats1)  #B, C, H, W  
        #virt_feats2
        virt_feats2 = self.norm(virt_feats2)
        #virt_feats2 = nn.functional.relu(virt_feats2)
        virt_feats2 = self.conv2(virt_feats2)
        virt_feats2 = self.dp2(virt_feats2)  #B, C, H, W  
        #bbox_feat1
        group_bbox_feat1 = bbox_feat + virt_feats1  #B, C, H, W  
        # group_bbox_feat1 = group_bbox_feat1.permute(0,2,1).reshape(B, C, H, W)
        group_bbox_feat1 = group_bbox_feat1.unsqueeze(2)  # (124,512,1,8,8)
        #bbox_feat2
        group_bbox_feat2 = bbox_feat + virt_feats2 #B, C, H, W  
        # group_bbox_feat2 = group_bbox_feat2.permute(0,2,1).reshape(B, C, H, W)
        group_bbox_feat2 = group_bbox_feat2.unsqueeze(2)  # (124,512,1,8,8)

        gropu_bbox_feats = [group_bbox_feat1, group_bbox_feat2]
        return gropu_bbox_feats

