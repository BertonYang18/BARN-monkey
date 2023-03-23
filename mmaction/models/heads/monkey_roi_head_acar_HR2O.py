# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class
import torch, torchvision
import math

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.roi_heads import monkey_StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

class HR2O_NL(nn.Module):
    def __init__(self, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(HR2O_NL, self).__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)  # 把hidden_dim个channel分为1个group来norm
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)  # x:(5,512,7,14)  conv_q ->  (5,512,7,14)  unsqueeze-> (5,1,512,7,14) 
        key = self.conv_k(x).unsqueeze(0)  # -> (1,5,512,7,14) 
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)  # (5,5,512,7,14)  --sum--> (5,5,7,14)  
        att = nn.Softmax(dim=1)(att)  # -> (5,5,7,14)  
        value = self.conv_v(x)  # (5,512,7,14)
        virt_feats = (att.unsqueeze(2) * value).sum(1)  # (5,5,1,7,14)*(5,512,7,14) ->(5,5,512,7,14)->(5,512,7,14)

        virt_feats = self.norm(virt_feats)  # ->(5,512,7,14)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)  # ->(5,512,7,14)
        virt_feats = self.dp(virt_feats)  # ->(5,512,7,14)

        x = x + virt_feats   # (5,512,7,14) + (5,512,7,14) = (5,512,7,14)
        return x

class ACARHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=16, dropout=0., bias=False,
                 reduce_dim=1024, hidden_dim=512, downsample='max2x2', depth=2,
                 kernel_size=3, mlp_1x1=False):
        super(ACARHead, self).__init__()

        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)
        # actor-context feature encoder
        # in_channels (int), out_channels (int), kernel_size (int or tuple), stride (int or tuple, optional)`
        self.conv_reduce = nn.Conv2d(width, reduce_dim, 1, bias=False)

        self.conv1 = nn.Conv2d(reduce_dim * 2, hidden_dim, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, bias=False)

        # down-sampling before HR2O
        assert downsample in ['none', 'max2x2']
        if downsample == 'none':
            self.downsample = nn.Identity()
        elif downsample == 'max2x2':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # high-order relation reasoning operator (HR2O_NL)
        layers = []
        for _ in range(depth):
            layers.append(HR2O_NL(hidden_dim, kernel_size, mlp_1x1))
        self.hr2o = nn.Sequential(*layers)

        # classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(reduce_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None


    # returns: outputs
    def forward(self, x, rois):
        # x:[f0, f1]     f0-(N,C_slow,T,H,W)   f1-(N,C_fast,T,H,W)
        feats = x
        if not isinstance(feats, (list, tuple)):
            feats = [feats]  # 单支路传来的x可能为 tensor 而不是 list(tensor)
        else:
            feats = feats
        
        n_batchsize = feats[0].shape[0]
        # temporal average pooling
        h, w = feats[0].shape[3:]
        # requires all features have the same spatial dimensions
        # f0: [6,2048,8,16,29]   f1:[6,256,32,16,29]      f0-(B,C_slow,T,H,W)   f1-(B,C_fast,T,H,W)
        # nn.AdaptiveAvgPool3d((1, h, w))(f)  ->  [6,2048,1,16,29]   [6,256,1,16,29]
        # .view-> [6,2048,16,29]    [6,256,16,29]
        # feats = [tensor(6,2048,16,29), tensor(6,256,16,29)]
        # cat -> tensor(6,2304,16,29)
        feats = [nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, f.shape[1], h, w) for f in feats]
        feats = torch.cat(feats, dim=1)
        # self.conv_reduce = nn.Conv2d(width, reduce_dim, 1, bias=False)   width = 2304  , reduce_dim=1024, kernel_size=1*1
        # -> (6,1024,16,29)
        feats = self.conv_reduce(feats)
        # 把bbox变成roi绝对坐标  roi:(16,29)

        num_rois = len(rois)
        rois[:, 1] = rois[:, 1] * w
        rois[:, 2] = rois[:, 2] * h
        rois[:, 3] = rois[:, 3] * w
        rois[:, 4] = rois[:, 4] * h
        # (17,5)
        rois = rois.detach()  # 新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，即requires_grad=False
        # feats: (6,1024,16,29)      rois:(17,5)     roi_spatial:7*7   -> # (17,1024,7,7) 
        roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))  
        # maxpool(7)(roi_feats) -> (17,1024,1,1) 
        # .view  -> (17,1024) 
        roi_feats = self.roi_maxpool(roi_feats).view(num_rois, -1)

        # rois=((0,bbox) (0,bbox) (1,bbox) (2,bbox))  batchsize=3 -- total 3 frames
        # roi_ids = (0, 2, 3, 4)   len(roi_ids)=batchsize+1 =4  
        # roi_ids[1]=2 means first frame  has 2 rois : (0,bbox) (0,bbox)
        # roi_ids[2]=3 means first two frame  have 3 rois : (0,bbox) (0,bbox) (1,bbox) 
        roi_ids = [0] * (n_batchsize + 1)
        rois_batch = rois[:,0].tolist()
        for i in range(n_batchsize): # 0 ~ batchsize - 1
            roi_ids[i+1] = rois_batch.count(i)  +  roi_ids[i]

        # sizes_before_padding = data['sizes_before_padding']
        high_order_feats = []
        for idx in range(n_batchsize):  # iterate over mini-batch
            n_rois = roi_ids[idx+1] - roi_ids[idx]  #当前帧的roi数量
            if n_rois == 0:
                continue
            # math.ceil: 向上取整
            # eff_h, eff_w = math.ceil(h * sizes_before_padding[idx][1]), math.ceil(w * sizes_before_padding[idx][0])
            # bg_feats = feats[idx][:, :eff_h, :eff_w]  # (6,1024,16,29)   ->  (1024,16,29)
            bg_feats = feats[idx]  # (1024,16,29)
            bg_feats = bg_feats.unsqueeze(0).repeat((n_rois, 1, 1, 1))  # (n_rois, 1024,16,29)
            actor_feats = roi_feats[roi_ids[idx]:roi_ids[idx+1]]  # (n_rois, 1024)  当前帧的rois
            tiled_actor_feats = actor_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats)  # (n_rois, 1024,16,29)
            interact_feats = torch.cat([bg_feats, tiled_actor_feats], dim=1)  # (n_rois, 2048,16,29)

            interact_feats = self.conv1(interact_feats)  # (n_rois, 512,16,29)
            interact_feats = nn.functional.relu(interact_feats)
            interact_feats = self.conv2(interact_feats)  # (H-3+2P)/S+1  ->  (n_rois, 512,14,27)
            interact_feats = nn.functional.relu(interact_feats)

            interact_feats = self.downsample(interact_feats)  #K=3,P=1,S=2    -> (n_rois, 512,7,14)

            interact_feats = self.hr2o(interact_feats)  # -> (n_rois, 512,7,14)
            interact_feats = self.gap(interact_feats)  # GAP-> (n_rois, 512,1,1)
            high_order_feats.append(interact_feats)
        # [batch0, ```,b6`]     b_i:  (n_rois_i, 512, 1, 1)  ->  (17, 512)
        high_order_feats = torch.cat(high_order_feats, dim=0).view(num_rois, -1)

        outputs = self.fc1(roi_feats)  # (num_rois, 1024)  ->  (num_rois, 512)  # Linear只对最后一个维度进行变换
        outputs = nn.functional.relu(outputs)
        outputs = torch.cat([outputs, high_order_feats], dim=1)  # (num_rois, 1024)

        if self.dp is not None:
            outputs = self.dp(outputs)
        cls_score = self.fc2(outputs)  # (num_rois, 1024)  -> (num_rois, num_classes)
        
        # We do not predict bbox, so return None
        bbox_pred = None
        return cls_score, bbox_pred, roi_feats


if mmdet_imported:

    @MMDET_HEADS.register_module()
    class monkey_AVARoIHead_acar_HR2O(monkey_StandardRoIHead):
        def __init__(self, width=2304, num_classes=17,
            bbox_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None):  #实例化时config参数先传入类内
            super(monkey_AVARoIHead_acar_HR2O, self).__init__(
                bbox_head=bbox_head,
                pretrained=pretrained,
                train_cfg=train_cfg,
                test_cfg=test_cfg)  #初始化父类
            self.acar_head = ACARHead(width=width, num_classes=num_classes)

        def _bbox_forward(self, x, rois, img_metas):
            """Defines the computation performed to get bbox predictions.

            Args:
                x (torch.Tensor): The input tensor.
                rois (torch.Tensor): The regions of interest.
                img_metas (list): The meta info of images

            Returns:
                dict: bbox predictions with features and classification scores.

            input:
                x:list(tensor)   [(9,2048,4,16,16),  (9,256,32,16,16)]
                rois:(60,5)     [(0,bbox_1), (0,bbox_2),```, (9,bbox_60)]
            """
            
            cls_score, bbox_pred, bbox_feat = self.acar_head(x, rois)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
            return bbox_results

        def _bbox_forward_train(self, x, sampling_results, gt_bboxes,
                                gt_labels, img_metas):
            """Run forward function and calculate loss for box head in
            training."""
            rois = bbox2roi([res.bboxes for res in sampling_results])  # 把list(Tensor)： 9*（n_bbox*4）转为tensor: m* 5==(num_batch, bbox*4)
            bbox_results = self._bbox_forward(x, rois, img_metas)  #得到不同roi对应的cls_score、对应的feat、bbox_pred -》bbox_results

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg)  # 得到不同roi对应的(label, label_weight) -> bbox_target    (shape(65*17) , shape(65,))
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)   # 该batch所有-65个roi对应的loss --  一个元素

            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results

        def simple_test(self,
                        x,
                        proposal_list,
                        img_metas,
                        proposals=None,
                        rescale=False):
            """Defines the computation performed for simple testing."""
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)
            #4*4    4*17  
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(  # 滤除 <thr的result ; 多标签/单标签
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                thr=self.test_cfg.action_thr)
            return [bbox_results]

        def simple_test_bboxes(self,
                               x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
            """Test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois, img_metas)
            cls_score = bbox_results['cls_score']  # 4*17   num of bbox=4 
            # bbox_results['bbox_feat'] :  (4,2304,1,8,8)
            img_shape = img_metas[0]['img_shape']
            crop_quadruple = np.array([0, 0, 1, 1])
            flip = False

            if 'crop_quadruple' in img_metas[0]:
                crop_quadruple = img_metas[0]['crop_quadruple']

            if 'flip' in img_metas[0]:
                flip = img_metas[0]['flip']
            #rois:4*5  5:[0, bbox*4]
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                img_shape,
                flip=flip,
                crop_quadruple=crop_quadruple,
                cfg=rcnn_test_cfg)

            return det_bboxes, det_labels  # 4*4    4*17
else:
    # Just define an empty class, so that __init__ can import it.
    @import_module_error_class('mmdet')
    class monkey_AVARoIHead_acar_HR2O:
        pass
