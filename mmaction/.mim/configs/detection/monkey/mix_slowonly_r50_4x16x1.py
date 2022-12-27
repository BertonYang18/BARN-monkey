#custom_classes = [1,3,5,6,7,8,10,11,12,13,14,15]
#custom_classes = [2,4,8,9,14,15,16]
custom_classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
num_classes = len(custom_classes) + 1

# model setting
model = dict(
    type='monkey_FastRCNN',
    #在mmaction/models/backbones/monkey_resnet3d_slowfast_acar.py中修改了默认layers
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        pretrained2d=False,
        lateral=False,
        num_stages=4,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        spatial_strides=(1, 2, 2, 1)),
    roi_head=dict(
        type='monkey_AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',  #return  roi_feat  4*C*
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True,
            with_global=False),
        bbox_head=dict(
            type='monkey_BBoxHeadAVA', #'monkey_BBoxHeadAVA_switch_group_linear'
            in_channels=2048,
            num_classes=num_classes,
            multilabel=True,
            dropout_ratio=0.5)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=dict(action_thr=0.002)))


dataset_type = 'MonkeyDataset'
data_root = '/data/ys/monkey/alldata/frames/all_frames'
anno_root = 'data/monkey/annotation_mix'

ann_file_train = f'{anno_root}/train.csv'
ann_file_val = f'{anno_root}/val.csv'

exclude_file_train = None
exclude_file_val = None

label_file = f'{anno_root}/monkey_action_list_index1.pbtxt'

proposal_file_train = (f'{anno_root}/gd_bbox_train.pkl')
proposal_file_val = f'{anno_root}/gd_bbox_val.pkl'

img_norm_cfg = dict(
    mean=[114.,114.,114.], std=[57.375,57.375,57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleMonkeyFrames', clip_len=16, frame_interval=1),  
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'entity_ids'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleMonkeyFrames', clip_len=16, frame_interval=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', 
        fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
        nested=True)
]

# test_pipeline = val_pipeline.copy()
# test_pipeline[-3:] = [    
#     dict(type='ToTensor', keys=['img', 'proposals']),
#     dict(type='ToDataContainer', 
#         fields=[dict(key=['proposals'], stack=False)]),
#     dict(
#         type='Collect',
#         keys=['img', 'proposals'],
#         meta_keys=['scores', 'img_shape'],
#         nested=True)
# ]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        person_det_score_thr=0.9,
        num_classes=num_classes,
        custom_classes=custom_classes,
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.9,
        num_classes=num_classes,
        custom_classes=custom_classes,
        data_prefix=data_root))
data['test'] = data['val']

optimizer = dict(type='SGD', lr=0.1125, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)
total_epochs = 30
checkpoint_config = dict(interval=5)
# workflow = []
workflow = [('train', 1)]
# workflow = [('val',1)]
evaluation = dict(interval=1, save_best='mAP@0.5IOU', gpu_collect=True)
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('/data/ys/mmaction/work_dirs/monkey/mix_no_seed_GPU3/'
            'mix_slowonly_r50_4x16x1_bs12_fps16_nopretrained')

# load_from = "checkpoint/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth"
# resume_from = '/data/ys/mmaction/work_dirs/monkey/mix_no_seed_GPU3/mix_slowonly_r50_4x16x1_bs12_fps16/best_mAP@0.5IOU_epoch_9.pth'
load_from = None
resume_from = None
find_unused_parameters = True
