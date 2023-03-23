# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import os
import os.path as osp
from pickle import TRUE
import time

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, get_dist_info)
from mmcv.runner.hooks import Fp16OptimizerHook

from ..core import (DistEvalHook, EvalHook, OmniSourceDistSamplerSeedHook,
                    OmniSourceRunner)
from ..datasets import build_dataloader, build_dataset
from ..utils import PreciseBNHook, get_root_logger
from .test import multi_gpu_test


def init_random_seed(seed=None, device='cuda', distributed=True):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
        distributed (bool): Whether to use distributed training.
            Default: True.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    if distributed:
        dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    if cfg.omnisource:
        # The option can override videos_per_gpu
        train_ratio = cfg.data.get('train_ratio', [1] * len(dataset))
        omni_videos_per_gpu = cfg.data.get('omni_videos_per_gpu', None)
        if omni_videos_per_gpu is None:
            dataloader_settings = [dataloader_setting] * len(dataset)
        else:
            dataloader_settings = []
            for videos_per_gpu in omni_videos_per_gpu:
                this_setting = cp.deepcopy(dataloader_setting)
                this_setting['videos_per_gpu'] = videos_per_gpu
                dataloader_settings.append(this_setting)
        data_loaders = [
            build_dataloader(ds, **setting)
            for ds, setting in zip(dataset, dataloader_settings)
        ]

    else:
        data_loaders = [
            build_dataloader(ds, **dataloader_setting) for ds in dataset
        ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)


    Runner = OmniSourceRunner if cfg.omnisource else EpochBasedRunner
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    # multigrid setting
    multigrid_cfg = cfg.get('multigrid', None)
    if multigrid_cfg is not None:
        from mmaction.utils.multigrid import LongShortCycleHook
        multigrid_scheduler = LongShortCycleHook(cfg)
        runner.register_hook(multigrid_scheduler)
        logger.info('Finish register multigrid hook')

        # subbn3d aggregation is HIGH, as it should be done before
        # saving and evaluation
        from mmaction.utils.multigrid import SubBatchNorm3dAggregationHook
        subbn3d_aggre_hook = SubBatchNorm3dAggregationHook()
        runner.register_hook(subbn3d_aggre_hook, priority='VERY_HIGH')
        logger.info('Finish register subbn3daggre hook')

    # precise bn setting
    if cfg.get('precise_bn', False):
        precise_bn_dataset = build_dataset(cfg.data.train)
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=1,  # save memory and time
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
        data_loader_precise_bn = build_dataloader(precise_bn_dataset,
                                                  **dataloader_setting)
        precise_bn_hook = PreciseBNHook(data_loader_precise_bn,
                                        **cfg.get('precise_bn'))
        runner.register_hook(precise_bn_hook, priority='HIGHEST')
        logger.info('Finish register precisebn hook')

    if distributed:
        if cfg.omnisource:
            runner.register_hook(OmniSourceDistSamplerSeedHook())
        else:
            runner.register_hook(DistSamplerSeedHook())

    val_dataloader = None
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook(val_dataloader, **eval_cfg) if distributed \
            else EvalHook(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()
    if cfg.omnisource:
        runner_kwargs = dict(train_ratio=train_ratio)
    
   
    ckt_pwd = '/data/ys/mmaction/work_dirs/monkey_BARN/interaction_2class_v2/swBB_after_change_interaction.pth'
    checkpoint = torch.load(ckt_pwd, map_location=lambda storage, loc: storage.cuda())

    model_dict = model.state_dict()  
    para_list = list(model.parameters()) 

    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)  


    net_dict = model.state_dict()
    net_list = list(net_dict)
    para_list = list(model.parameters()) 

    for i in range(len(net_dict),0,-1):
        ind = i - 1 # ind~ len(net_dict)-1 -> 0    i~len(net_dict) -> 1
        name = net_list[ind]
        if 'weight' not in name and 'bias' not in name:
            del net_list[ind]
    '''freezing FEM ackbone'''
    # start_num = 0
    # end_num = 1
    # just = True
    # for i,n in enumerate(net_list):
    #     if 'backbone' in n and 'switch' not in n:
    #         end_num = i
    #         if just and i>0 :
    #             if  'backbone' not in net_list[i-1]:
    #                 just = False
    #                 start_num = i
    #     # if 'head'  in n:
    #     #     end_num = i      
    # # print(net_list[start_num:end_num+1])   #len==662
    # # print(para_list[start_num:end_num+1])  #len==332
    # #对模型的start_num~end_num参数进行冻结
    # for i,p in enumerate(model.parameters()):
    #     if start_num <= i <= end_num:
    #         p.requires_grad = False
    #         #print(p)
    
    
    '''freezing BAM backbone'''
    # start_num = 0
    # end_num = len(net_list) - 1
    start_num = 0
    end_num = 1
    just = True
    for i,n in enumerate(net_list):
        if 'switch_backbone'  in n:
            end_num = i
            if just and i>0 :
                if  'switch_backbone' not in net_list[i-1]:
                    just = False
                    start_num = i
        if 'head'  in n:  # freeze all layers  (backbone & head)
            end_num = i      
    # print(net_list[start_num:end_num+1])   #len==662
    # print(para_list[start_num:end_num+1])  #len==332
    for i,p in enumerate(model.parameters()):
        if start_num <= i <= end_num:
            p.requires_grad = False
            #print(p)
        # if 658 <= i <= 659:  #冻结switch_head
        #     # print(p)
        #     p.requires_grad = False
    # print(para_list[start_num:end_num+1]) 

    
    data_loaders = {'train':data_loaders[0], 'val':val_dataloader}
    if distributed:
        dist.barrier()
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, **runner_kwargs)
    if distributed:
        dist.barrier()
    time.sleep(5)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
            ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
            if len(ckpt_paths) == 0:
                runner.logger.info('Warning: test_best set, but no ckpt found')
                test['test_best'] = False
                if not test['test_last']:
                    return
            elif len(ckpt_paths) > 1:
                epoch_ids = [
                    int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths
                ]
                best_ckpt_path = ckpt_paths[np.argmax(epoch_ids)]
            else:
                best_ckpt_path = ckpt_paths[0]
            if best_ckpt_path:
                best_ckpt_path = osp.join(cfg.work_dir, best_ckpt_path)

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        gpu_collect = cfg.get('evaluation', {}).get('gpu_collect', False)
        tmpdir = cfg.get('evaluation', {}).get('tmpdir',
                                               osp.join(cfg.work_dir, 'tmp'))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))

        test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

        names, ckpts = [], []

        if test['test_last']:
            names.append('last')
            ckpts.append(None)
        if test['test_best'] and best_ckpt_path is not None:
            names.append('best')
            ckpts.append(best_ckpt_path)

        for name, ckpt in zip(names, ckpts):
            if ckpt is not None:
                runner.load_checkpoint(ckpt)
            #format: outputs: List    len==num of img_key 
            #           output[i]:List. i<num of img_key   len==16 num of action
            #               output[i][a]:array. a=0~15   shape: n*5 :  num of bbox * (bboxes*4, score)
            outputs = multi_gpu_test(runner.model, test_dataloader, tmpdir,
                                     gpu_collect)
            rank, _ = get_dist_info()
            if rank == 0:
                out = osp.join(cfg.work_dir, f'{name}_pred.csv')
                test_dataset.dump_results(outputs, out)

                eval_cfg = cfg.get('evaluation', {})
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect',
                        'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
                ]:
                    eval_cfg.pop(key, None)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                runner.logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    runner.logger.info(f'{metric_name}: {val:.04f}')
