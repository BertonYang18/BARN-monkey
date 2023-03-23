# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import socket
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)

def get_env_variable(variables,  default=None):
    for candidate in variables:
        if candidate in os.environ:
            return os.environ[candidate]
    return default

def init_distributed(local_rank, args, backend='nccl'):
    if args.nnodes is not None:
        n_nodes = args.nnodes
    else:
        n_nodes = int(get_env_variable(['SLURM_NTASKS', 'MV2_COMM_WORLD_SIZE', 'PMI_SIZE'], default=1))
    if args.node_rank is not None:
        node_id = args.node_rank
    else:
        node_id = int(get_env_variable(['SLURM_PROCID', 'MV2_COMM_WORLD_RANK', 'PMI_RANK'], default=0))

    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['MASTER_ADDR'] = str(args.master_addr)
    
    world_size = n_nodes * args.nproc_per_node
    rank = node_id * args.nproc_per_node + local_rank
    dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
    print('[rank {:04d}]: distributed init: world_size={}, local_rank={}'.format(rank, world_size, local_rank), flush=True)
    
    num_gpus = torch.cuda.device_count() 
    torch.cuda.set_device(local_rank%num_gpus)
    
    return rank, world_size

'''
BARN
configs/detection/monkey_interaction/mix_r50_4x16x1_20e_ava_rgb_custom.py

BAM
configs/detection/monkey_interaction/mix_r50_4x16x1_20e_ava_rgb_custom_BAM_2class.py

slowonly
configs/detection/monkey_interaction/mix_slowonly_r50_4x16x1.py

acrn
configs/detection/monkey_interaction/mix_r50_4x16x1_20e_ava_rgb_custom_acrn.py
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', default="configs/detection/monkey_interaction/mix_r50_4x16x1_20e_ava_rgb_custom.py",help='train config file path')
    parser.add_argument('--master_port', type=int, default=19002)
    parser.add_argument('--master_addr', type=str, default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument('--nproc_per_node', type=int, default=2)
    parser.add_argument('--nnodes', type=int, default=None)
    parser.add_argument('--node_rank', type=int, default=None)
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        default=True,
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs="+",
        default=[0,1],
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args



def main(local_rank, args):
    
    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)  # 替换cfg中某个参数

    # set multi-process settings
    setup_multi_processes(cfg)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if args.gpu_ids is not None or args.gpus is not None:
            warnings.warn(
                'The Args `gpu_ids` and `gpus` are only used in non-distributed '
                'mode and we highly encourage you to use distributed mode, i.e., '
                'launch training with dist_train.sh. The two args will be '
                'deperacted.')
            if args.gpu_ids is not None:
                # warnings.warn(
                #     'Non-distributed training can only use 1 gpu now. We will '
                #     'use the 1st one in gpu_ids. ')
                cfg.gpu_ids = [args.gpu_ids[0]]
            elif args.gpus is not None:
                warnings.warn('Non-distributed training can only use 1 gpu now. ')
                cfg.gpu_ids = range(1)
    else:
        distributed = True
        # init_dist(args, **cfg.dist_params)
        # _, world_size = get_dist_info()
        # cfg.gpu_ids = range(world_size)
        rank, world_size = init_distributed(local_rank, args)
        cfg.gpu_ids = range(world_size)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if rank == 0:
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])
        if args.resume_from is not None:
            cfg.resume_from = args.resume_from


    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    if rank == 0:
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if rank == 0:
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    meta['env_info'] = env_info

    if rank == 0:
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config: {cfg.pretty_text}')

    # set random  s
    seed = init_random_seed(args.seed, distributed=distributed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    if rank == 0:
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

   
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    if cfg.omnisource:
        # If omnisource flag is set, cfg.data.train should be a list
        assert isinstance(cfg.data.train, list)
        datasets = [build_dataset(dataset) for dataset in cfg.data.train]
    else:
        datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        # For simplicity, omnisource is not compatible with val workflow,
        # we recommend you to use `--validate`
        assert not cfg.omnisource
        if args.validate:
            warnings.warn('val workflow is duplicated with `--validate`, '
                          'it is recommended to use `--validate`. see '
                          'https://github.com/open-mmlab/mmaction2/pull/123')
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    train_model(
        model,
        datasets,  
        cfg,
        distributed=distributed,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    args = parse_args()
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.nproc_per_node)

