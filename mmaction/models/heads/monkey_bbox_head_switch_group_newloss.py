# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.core.bbox import bbox_target

try:
    from mmdet.models.builder import HEADS as MMDET_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

# Resolve cross-entropy function to support multi-target in Torch < 1.10
#   This is a very basic 'hack', with minimal functionality to support the
#   procedure under prior torch versions
from packaging import version as pv

if pv.parse(torch.__version__) < pv.parse('1.10'):

    def cross_entropy_loss(input, target, reduction='None'):
        input = input.log_softmax(dim=-1)  # Compute Log of Softmax
        loss = -(input * target).sum(dim=-1)  # Compute Loss manually
        if reduction.lower() == 'mean':
            return loss.mean()
        elif reduction.lower() == 'sum':
            return loss.sum()
        else:
            return loss
else:
    cross_entropy_loss = F.cross_entropy


class monkey_BBoxHeadAVA_switch_group_newloss(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively.

    Args:
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
        in_channels (int): The number of input channels. Default: 2048.
        focal_alpha (float): The hyper-parameter alpha for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 1.
        focal_gamma (float): The hyper-parameter gamma for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 0.
        num_classes (int): The number of classes. Default: 81.
        dropout_ratio (float): A float in [0, 1], indicates the dropout_ratio.
            Default: 0.
        dropout_before_pool (bool): Dropout Feature before spatial temporal
            pooling. Default: True.
        topk (int or tuple[int]): Parameter for evaluating Top-K accuracy.
            Default: (3, 5)
        multilabel (bool): Whether used for a multilabel task. Default: True.
    """

    def __init__(
            self,
            temporal_pool_type='avg',
            spatial_pool_type='max',
            in_channels=2048,
            focal_gamma=0.,
            focal_alpha=1.,
            custom_classes=None,
            num_classes=81,  # First class reserved (BBox as pos/neg)
            dropout_ratio=0,
            dropout_before_pool=True,
            topk=(3,5),
            multilabel=True):

        super(monkey_BBoxHeadAVA_switch_group_newloss, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.custom_classes = custom_classes
        self.num_classes_all = 20

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk, )
        elif isinstance(topk, tuple):
            assert all([isinstance(k, int) for k in topk])
            self.topk = topk
        else:
            raise TypeError('topk should be int or tuple[int], '
                            f'but get {type(topk)}')
        # Class 0 is ignored when calculating accuracy,
        #      so topk cannot be equal to num_classes.
        assert all([k < num_classes for k in self.topk])


        # Pool by default
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        #self.cls_score = torch.tensor([100.]*self.num_classes_all).reshape(1,self.num_classes_all) #->(1, num_classes)
        slow_classes = [1,  3,  5,6,7,8, 10,11,12,13,14,15,           19]
        fast_classes = [  2,  4,      8,9,           14,15,16,17, 18,   ]
        self.slow_classes, self.fast_classes = [], []
        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            #_, class_whitelist = read_labelmap(open(label_file))
            #assert set(custom_classes).issubset(class_whitelist)
            #self.custom_classes = tuple([0] + custom_classes)

            for cls in custom_classes:
                if cls in slow_classes:
                    self.slow_classes.append(cls)
                if cls in fast_classes:
                    self.fast_classes.append(cls)
            self.slow_classes = tuple([0] + self.slow_classes)
            self.fast_classes = tuple([0] + self.fast_classes)
        else:
            self.slow_classes = tuple([0] + slow_classes)
            self.fast_classes = tuple([0] + fast_classes)
        self.num_classes_slow = len(self.slow_classes)
        self.num_classes_fast = len(self.fast_classes)
        #self.fc_cls = nn.Linear(in_channels, num_classes)
        self.fc_cls1 = nn.Linear(in_channels, self.num_classes_slow)
        self.fc_cls2 = nn.Linear(in_channels, self.num_classes_fast)
        self.debug_imgs = None
        #8 14 15
        self.mix = nn.Linear(1, 1)
    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def delete_extra_column(self, cls_score, c):
        #cls_score为二维tensor   (B, num_class)
        before_c = cls_score[:, 0:c]
        after_c = cls_score[:, c+1:]
        
        return torch.cat((before_c, after_c), dim=1)

    def forward(self, x, status):
        x1, x2 = x 
        #slow
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x1 = self.dropout(x1)
        x1 = self.temporal_pool(x1)
        x1 = self.spatial_pool(x1)
        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x1 = self.dropout(x1)
        
        x1 = x1.view(x1.size(0), -1)
        #fast
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x2 = self.dropout(x2)
        x2 = self.temporal_pool(x2)
        x2 = self.spatial_pool(x2)
        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x2 = self.dropout(x2)
        x2 = x2.view(x2.size(0), -1)

        B, _ = x1.size()
        #定义时若为[100]，默认torch.int，之后cls_score[:, actual_label] = cls_score_slow[:, i]会自动四舍五入
        cls_score = torch.tensor([-1000.]*self.num_classes_all).reshape(1,self.num_classes_all).repeat(B, 1).cuda() #->(B, num_classes)
        cls_score_slow = self.fc_cls1(x1)
        cls_score_fast = self.fc_cls2(x2)
        for i in range(self.num_classes_slow):
            actual_label = self.slow_classes[i]
            cls_score[:, actual_label] = cls_score_slow[:, i]
        for i in range(1, self.num_classes_fast):
            actual_label = self.fast_classes[i]
            # cls_score[:, actual_label] = cls_score_fast[:, i]
            if actual_label not in [8, 14, 15]:
                cls_score[:, actual_label] = cls_score_fast[:, i]
            else:
                # r = torch.rand(B,1) > 0.5
                # sr = torch.tensor(list(map(float, r))).cuda()
                if actual_label == 8:
                    slow = cls_score_slow[:, 6:7]
                    fast = cls_score_fast[:, 3:4]
                elif actual_label == 14:
                    slow = cls_score_slow[:, 11:12]
                    fast = cls_score_fast[:, 5:6]
                else:  # actual_label == 15
                    slow = cls_score_slow[:, 12:13]
                    fast = cls_score_fast[:, 6:7]

                out = slow + fast

                # out = torch.cat([slow, fast], dim=1)

                # out = self.mix(out)

                cls_score[:, actual_label] = out.squeeze(1)




        extra_column = []
        #得到要删除的列
        for c in range(self.num_classes_all-1, 0, -1):
            if cls_score[0, c] == -1000:
                extra_column.append(c)
        #执行删除
        for c in extra_column:
            cls_score = self.delete_extra_column(cls_score, c)


        # We do not predict bbox, so return None
        return cls_score, None

    @staticmethod
    def get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]  # 标注的bbox   list[Tensor]  len(list)==batch  tensor.shape(n_bbox, 4)
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, rcnn_train_cfg)
        return cls_reg_targets

    @staticmethod
    def get_recall_prec(pred_vec, target_vec):
        """Computes the Recall/Precision for both multi-label and single label
        scenarios.

        Note that the computation calculates the micro average.

        Note, that in both cases, the concept of correct/incorrect is the same.
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1 - for
                single label it is expected that only one element is on (1)
                although this is not enforced.
        """
        correct = pred_vec & target_vec
        recall = correct.sum(1) / target_vec.sum(1).float()  # Enforce Float
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    @staticmethod
    def topk_to_matrix(probs, k):
        """Converts top-k to binary matrix."""
        topk_labels = probs.topk(k, 1, True, True)[1]
        topk_matrix = probs.new_full(probs.size(), 0, dtype=torch.bool)
        for i in range(probs.shape[0]):
            topk_matrix[i, topk_labels[i]] = 1
        return topk_matrix

    def topk_accuracy(self, pred, target, thr=0.5):
        """Computes the Top-K Accuracies for both single and multi-label
        scenarios."""
        # Define Target vector:
        target_bool = target > 0.5

        # Branch on Multilabel for computing output classification
        if self.multilabel:
            pred = pred.sigmoid()
        else:
            pred = pred.softmax(dim=1)

        # Compute at threshold (K=1 for single)
        if self.multilabel:
            pred_bool = pred > thr
        else:
            pred_bool = self.topk_to_matrix(pred, 1)
        recall_thr, prec_thr = self.get_recall_prec(pred_bool, target_bool)

        # Compute at various K
        recalls_k, precs_k = [], []
        for k in self.topk:
            pred_bool = self.topk_to_matrix(pred, k)
            recall, prec = self.get_recall_prec(pred_bool, target_bool)
            recalls_k.append(recall)
            precs_k.append(prec)

        # Return all
        return recall_thr, prec_thr, recalls_k, precs_k


    def monkey_pose_softmax_func(self, logits):
        #此时已经删去了背景类
        assert len(logits[0,:]) <= 16, "最终类别数量不为16"
        #0~15:  7-进食  13-饮水  14-抓食 是可能与其他动作组合的
        pose_logits = torch.cat([logits[:, :7], logits[:, 8:13], logits[:, 15:]], dim=1)  #[0 1 2 3 4 5 6    8 9 10 11 12     15 ]
        interact_logits = torch.cat([logits[:, 7:8], logits[:, 13:14], logits[:, 14:15]] , dim=1)  #[7 13 14]

        pose_logits = nn.Softmax(dim=1)(pose_logits)  #姿态行为
        interact_logits = nn.Sigmoid()(interact_logits)  #交互行为   
        logits = torch.cat([pose_logits[:, :7], interact_logits[:,0:1] ,pose_logits[:, 7:12], logits[:, 1:3],pose_logits[:, 12:]], dim=1)
        
        logits = torch.clamp(logits, min=1e-6, max=1.)
        return logits

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True):

        losses = dict()
        # Only use the cls_score
        if cls_score is not None:
            labels = labels[:, 1:]  # Get valid labels (ignore first one)   65*16
            pos_inds = torch.sum(labels, dim=-1) > 0  # tensor(65)   有6个False,可能因为first one label为1,被删除
            cls_score = cls_score[pos_inds, 1:]  # (59,16)
            labels = labels[pos_inds]  # (59,16)

            # Compute First Recall/Precisions
            #   This has to be done first before normalising the label-space.
            recall_thr, prec_thr, recall_k, prec_k = self.topk_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]

            # If Single-label, need to ensure that target labels sum to 1: ie
            #   that they are valid probabilities.
            if not self.multilabel:
                labels = labels / labels.sum(dim=1, keepdim=True)

            # Select Loss function based on single/multi-label
            #   NB. Both losses auto-compute sigmoid/softmax on prediction
            #cls_score = self.monkey_pose_softmax_func(cls_score)
            if self.multilabel:
                loss_func = F.binary_cross_entropy_with_logits
                #loss_func = F.binary_cross_entropy
            else:
                loss_func = cross_entropy_loss

            # Compute loss
            #loss = loss_func(cls_score, labels, reduction='none')  #59*16
            loss = loss_func(cls_score, labels)  #59*16
            pt = torch.exp(-loss)
            F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss  # 59*16
            losses['loss_action_cls'] = torch.mean(F_loss)  # torch.Size([])   只有一个元素

        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        # Handle Multi/Single Label
        if cls_score is not None:
            if self.multilabel:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(dim=-1)
        else:
            scores = None

        bboxes = rois[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_shape
        if flip:
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        return bboxes, scores


if mmdet_imported:
    MMDET_HEADS.register_module()(monkey_BBoxHeadAVA_switch_group_newloss)
