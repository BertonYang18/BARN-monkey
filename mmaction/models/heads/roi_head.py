# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.roi_heads import StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class AVARoIHead(StandardRoIHead):

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
            bbox_feat, global_feat = self.bbox_roi_extractor(x, rois)

            if self.with_shared_head:
                bbox_feat = self.shared_head(
                    bbox_feat,
                    feat=global_feat,
                    rois=rois,
                    img_metas=img_metas)

            cls_score, bbox_pred = self.bbox_head(bbox_feat)

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
    class AVARoIHead:
        pass
