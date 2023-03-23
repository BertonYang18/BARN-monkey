# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import build_shared_head
from .monkey_switch_backbone import monkey_switch_ResNet3dSlowFast
from .monkey_switch_head import monkey_switch_Head


class monkey_switch_group_QKV_three(nn.Module):
    def __init__(self, status_dim=2, input_dim=2304, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(monkey_switch_group_QKV_three, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  #保持dim不变
        
        '''第一阶段'''
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
        self.conv11 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)
        self.conv12 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)

        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)  # 把hidden_dim个channel分为1个group来norm
        self.dp = nn.Dropout(0.2)
        '''第二阶段'''
        #q
        self.conv_q21 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)  #***hidden_dim
        self.conv_q22 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)  #***hidden_dim
        #k,v
        self.conv_k21 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v21 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k22 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v22 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        # virt_feat
        self.conv21 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)
        self.conv22 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)
        '''第三阶段'''
        #q
        self.conv_q31 = nn.Conv2d(status_dim, hidden_dim, kernel_size, padding=padding, bias=False)  #***hidden_dim
        self.conv_q32 = nn.Conv2d(status_dim, hidden_dim, kernel_size, padding=padding, bias=False)  #***hidden_dim
        #k,v
        self.conv_k31 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v31 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k32 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v32 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        # virt_feat
        self.conv31 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)
        self.conv32 = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)

    def forward(self, status, rois, bbox_feat):
        #bbox_feat:(124,2304,1,8,8)
        bbox_feat = bbox_feat.squeeze(2)  # -> (124,2304,8,8)
        bbox_feat = self.conv_reduce(bbox_feat)  # -> (124,512,8,8)
        '''第一阶段'''
        #status:(124,2)
        B, _, H, W = bbox_feat.size()  # C == hidden_dim
        status = torch.cat((status, rois[:, 1:]), dim=1)  #(124,2) + (124,4) -> (124, 6)
        #status = rois[:, 1:]
        status = status.unsqueeze(2).unsqueeze(2).expand(B,self.status_dim,H,W)  # ->(124,6,8,8)
        
        #q1 q2
        query1 = self.conv_q1(status).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
        query2 = self.conv_q2(status).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
        #k1, v1   
        key1 = self.conv_k1(bbox_feat).unsqueeze(0)  # -> (1，124,512,8,8)
        att1 = (query1 * key1).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
        att1 = nn.Softmax(dim=1)(att1)
        value1 = self.conv_v1(bbox_feat)  # (124,512,8,8)
        virt_feats1 = (att1.unsqueeze(2) * value1).sum(1)   # (124,512,8,8)
        #k2, v2
        key2 = self.conv_k2(bbox_feat).unsqueeze(0)  # -> (1，124,512,8,8)
        att2 = (query2 * key2).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
        att2 = nn.Softmax(dim=1)(att2)
        value2 = self.conv_v2(bbox_feat)  # (124,512,8,8)
        virt_feats2 = (att2.unsqueeze(2) * value2).sum(1)   # (124,512,8,8)
        #virt_feats1
        virt_feats1 = self.norm(virt_feats1)
        virt_feats1 = nn.functional.relu(virt_feats1)
        virt_feats1 = self.conv11(virt_feats1)
        virt_feats1 = self.dp(virt_feats1)
        #virt_feats2
        virt_feats2 = self.norm(virt_feats2)
        virt_feats2 = nn.functional.relu(virt_feats2)
        virt_feats2 = self.conv12(virt_feats2)
        virt_feats2 = self.dp(virt_feats2)
        #bbox_feat1  慢
        group_bbox_feat1 = bbox_feat + virt_feats1 # (124,512,8,8) +=  (124,512,8,8)
        #bbox_feat2  快
        group_bbox_feat2 = bbox_feat + virt_feats2 # (124,512,8,8) +=  (124,512,8,8)


        '''第二阶段'''
        #q1 q2
        query21 = self.conv_q21(group_bbox_feat1).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
        query22 = self.conv_q22(group_bbox_feat2).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
        #k1, v1   
        key21 = self.conv_k21(group_bbox_feat1).unsqueeze(0)  # -> (1，124,512,8,8)
        att21 = (query21 * key21).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
        att21 = nn.Softmax(dim=1)(att21)
        value21 = self.conv_v21(group_bbox_feat1)  # (124,512,8,8)
        virt_feats21 = (att21.unsqueeze(2) * value21).sum(1)   # (124,512,8,8)
        #k2, v2
        key22 = self.conv_k22(group_bbox_feat2).unsqueeze(0)  # -> (1，124,512,8,8)
        att22 = (query22 * key22).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
        att22 = nn.Softmax(dim=1)(att22)
        value22 = self.conv_v22(group_bbox_feat2)  # (124,512,8,8)
        virt_feats22 = (att22.unsqueeze(2) * value22).sum(1)   # (124,512,8,8)
        #virt_feats1
        virt_feats21 = self.norm(virt_feats21)
        virt_feats21 = nn.functional.relu(virt_feats21)
        virt_feats21 = self.conv21(virt_feats21)
        virt_feats21 = self.dp(virt_feats21)
        #virt_feats2
        virt_feats22 = self.norm(virt_feats22)
        virt_feats22 = nn.functional.relu(virt_feats22)
        virt_feats22 = self.conv22(virt_feats22)
        virt_feats22 = self.dp(virt_feats22)
        #bbox_feat1  慢
        group_bbox_feat_second1 = group_bbox_feat1 + virt_feats21 # (124,512,8,8) +=  (124,512,8,8)
        #bbox_feat2  快
        group_bbox_feat_second2 = group_bbox_feat2 + virt_feats22 # (124,512,8,8) +=  (124,512,8,8)


        '''第三阶段'''
        #q1 q2
        query31 = self.conv_q31(status).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
        query32 = self.conv_q32(status).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
        #k1, v1   
        key31 = self.conv_k31(group_bbox_feat_second1).unsqueeze(0)  # -> (1，124,512,8,8)
        att31 = (query31 * key31).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
        att31 = nn.Softmax(dim=1)(att31)
        value31 = self.conv_v31(group_bbox_feat_second1)  # (124,512,8,8)
        virt_feats31 = (att31.unsqueeze(2) * value31).sum(1)   # (124,512,8,8)
        #k2, v2
        key32 = self.conv_k32(group_bbox_feat_second2).unsqueeze(0)  # -> (1，124,512,8,8)
        att32 = (query32 * key32).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
        att32 = nn.Softmax(dim=1)(att32)
        value32 = self.conv_v32(group_bbox_feat_second2)  # (124,512,8,8)
        virt_feats32 = (att32.unsqueeze(2) * value32).sum(1)   # (124,512,8,8)
        #virt_feats1
        virt_feats31 = self.norm(virt_feats31)
        virt_feats31 = nn.functional.relu(virt_feats31)
        virt_feats31 = self.conv31(virt_feats31)
        virt_feats31 = self.dp(virt_feats31)
        #virt_feats2
        virt_feats32 = self.norm(virt_feats32)
        virt_feats32 = nn.functional.relu(virt_feats32)
        virt_feats32 = self.conv32(virt_feats32)
        virt_feats32 = self.dp(virt_feats32)
        #bbox_feat1  慢
        group_bbox_feat_third1 = group_bbox_feat_second1 + virt_feats31 # (124,512,8,8) +=  (124,512,8,8)
        #bbox_feat2  快
        group_bbox_feat_third2 = group_bbox_feat_second2 + virt_feats32 # (124,512,8,8) +=  (124,512,8,8)



        group_bbox_feat_third1 = group_bbox_feat_third1.unsqueeze(2)  # (124,512,1,8,8)
        group_bbox_feat_third2 = group_bbox_feat_third2.unsqueeze(2)  # (124,512,1,8,8)
        group_bbox_feats_third= [group_bbox_feat_third1, group_bbox_feat_third2]
        return group_bbox_feats_third

