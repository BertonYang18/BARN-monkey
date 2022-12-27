# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch.nn as nn
from mmcv.runner import BaseModule
import torch
from ..builder import build_shared_head
from .monkey_switch_backbone import monkey_switch_ResNet3dSlowFast
from .monkey_switch_head import monkey_switch_Head


class monkey_switch_QKV(nn.Module):
    def __init__(self, status_dim=2, input_dim=2304, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(monkey_switch_QKV, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  #保持dim不变
        self.conv_reduce = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # status, bbox_featx
        self.status_dim = status_dim
        self.conv_q = nn.Conv2d(self.status_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        #q,k,v
        #self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        # virt_feat
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else kernel_size, padding=0 if mlp_1x1 else padding, bias=False)

        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)  # 把hidden_dim个channel分为1个group来norm
        self.dp = nn.Dropout(0.2)

    def forward(self, status, rois, bbox_featx):
        #bbox_featx:(124,2304,1,8,8)
        bbox_featx = bbox_featx.squeeze(2)  # -> (124,2304,8,8)
        bbox_featx = self.conv_reduce(bbox_featx)

        #status:(124,2)
        B, _, H, W = bbox_featx.size()  # C == hidden_dim
        status = torch.cat((status[:,0].reshape(B,1), rois[:, 1:]), dim=1)  #(124,1) + (124,4) -> (124, 5)
        status = status.unsqueeze(2).unsqueeze(2).expand(B,self.status_dim,H,W)  # ->(124,5,8,8)
        
        query = self.conv_q(status).unsqueeze(1)  # ->(124,512,8,8) ->(124,1,512,8,8)
        # q
        #query = self.conv_q(status).unsqueeze(1)

        # k, v    
        key = self.conv_k(bbox_featx).unsqueeze(0)  # -> (1，124,512,8,8)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5) # (124，124,8,8)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(bbox_featx)  # (124,512,8,8)
        virt_feats = (att.unsqueeze(2) * value).sum(1)   # (124,512,8,8)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)

        bbox_featx = bbox_featx + virt_feats # (124,512,8,8) +=  (124,512,8,8)
        bbox_featx = bbox_featx.unsqueeze(2)  # (124,512,1,8,8)
        return bbox_featx

