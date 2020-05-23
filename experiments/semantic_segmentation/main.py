import argparse
import numpy as np
import random
import os
import torch
import visdom
from torch import nn

from spherical_distortion.util import distributed as distr_util
from spherical_distortion.util import OpMode, make_repeatable

from ss.config_defaults import _C as cfg
from ss.data import build_dataloader
from ss.models import build_model, get_checkpoint_path
from ss.models import build_optimizer, build_criterion, build_scheduler
from ss.engine import Engine


def parse_args():
    parser = argparse.ArgumentParser(description='Tangent planes in PyTorch')
    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        help='Path to configuration file.',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Launch in distributed mode. Please use run_dist_train tool.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate the trained model.')
    parser.add_argument(
        'overwrite_args',
        help='Overwrite args from config_file through the command line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def check_arguments(cfg, args):
    assert os.path.exists(cfg.DATA_ROOT), 'Data root folder does not exist.'
    assert cfg.DATASET in ['synthia', 'stanford'], 'Dataset not supported.'
    assert cfg.MODEL_TYPE in ['zhangunet', 'resnet101', 'hexunet']
    assert cfg.SCHEDULER in ['step', 'multistep', 'thirdparty']
    if cfg.DATA_FORMAT == 'pano' and cfg.EVAL_PER_PATCH != 0:
        assert 20 * (
            4**cfg.BASE_ORDER
        ) % cfg.EVAL_PER_PATCH == 0, 'EVAL_PER_PATCH should be a factor of 20(4^b)'


if __name__ == '__main__':
    # ================================
    # Parse arguments
    # ================================
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.overwrite_args)
    cfg.freeze()
    check_arguments(cfg, args)
    distr_util.dprint('Running with config:')
    distr_util.dprint(cfg.dump())

    # Experiment name and folders
    suffix = '' if not len(cfg.EXPERIMENT_SUFFIX) else '-' + args.suffix
    experiment_name = '{}-{}-patches-{}-l{}b{}-fold{}-bs{}{}'.format(
        cfg.DATASET, cfg.DATA_FORMAT, cfg.MODEL_TYPE, cfg.SAMPLE_ORDER,
        cfg.BASE_ORDER, cfg.FOLD, cfg.BATCH_SIZE_PER_GPU,
        suffix) if not len(cfg.EXPERIMENT_NAME) else cfg.EXPERIMENT_NAME

    checkpoint_dir = os.path.join(cfg.CHECKPOINT_ROOT, experiment_name)
    checkpoint_path, load_weights_only = get_checkpoint_path(
        checkpoint_dir, args.evaluate, cfg.START_EPOCH, cfg.MODEL_PATH,
        cfg.LOAD_WEIGHTS_ONLY)

    # ================================
    # Repeatability
    # ================================
    make_repeatable(0)

    # ================================
    # Initialize devices
    # ================================
    if args.distributed:
        distr_util.initialize(args.local_rank)
    if not cfg.GPU_LIST:
        device_ids = list(range(torch.cuda.device_count()))
    else:
        device_ids = cfg.GPU_LIST
    num_gpus = len(device_ids)
    distr_util.dprint('Using devices {}'.format(device_ids))

    # ================================
    # Dataset info
    # ================================
    train_dataloader = None
    if not args.evaluate:
        train_dataloader = build_dataloader(
            cfg, num_gpus, args.distributed, is_train=True)
    test_dataloader = build_dataloader(
        cfg, num_gpus, args.distributed, is_train=False)
    num_classes = test_dataloader.dataset.num_classes
    H, W = test_dataloader.dataset.pano_shape
    scale_factor = test_dataloader.dataset.scale_factor
    distr_util.dprint('scale_factor=', scale_factor)
    image_shape = (int(H * scale_factor), int(W * scale_factor))
    norm_stats = (test_dataloader.dataset.mean, test_dataloader.dataset.std)

    # ================================
    # Model info
    # ================================
    model = build_model(cfg, num_classes)
    torch.backends.cudnn.benchmark = True

    if not args.distributed:
        model = torch.nn.DataParallel(
            model,
            device_ids=device_ids,
        )
        model = model.to(cfg.DEVICE)
    else:
        model = model.to(cfg.DEVICE)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    # ================================
    # Set up training utilities
    # ================================
    optimizer = None
    criterion = None
    scheduler = None
    if not args.evaluate:
        optimizer = build_optimizer(cfg, model)
        criterion = build_criterion(cfg)
        scheduler = build_scheduler(cfg, optimizer)

    # Visdom visualization
    vis = visdom.Visdom(server=cfg.VISDOM.SERVER, env=experiment_name) \
        if args.local_rank == 0 and not args.evaluate and cfg.VISDOM.USE_VISDOM else None

    # Instantiate training engine
    engine = Engine(
        network=model,
        name=experiment_name,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.NUM_EPOCHS,
        validation_freq=cfg.VALIDATION_FREQ,
        checkpoint_freq=cfg.CHECKPOINT_FREQ,
        visualization_freq=cfg.VIZ_FREQ,
        display_samples=cfg.VISDOM.DISPLAY_SAMPLES,
        evaluation_sample_freq=cfg.SAMPLE_FREQ,
        logfile=cfg.LOGFILE if cfg.LOGFILE else None,
        checkpoint_root=cfg.CHECKPOINT_ROOT,
        sample_root=cfg.SAMPLE_ROOT,
        op_mode=OpMode(cfg.OP_MODE),
        distributed=args.distributed,
        device=cfg.DEVICE,
        visdom=vis,
        image_shape=image_shape,
        base_order=cfg.BASE_ORDER,
        sample_order=cfg.SAMPLE_ORDER,
        data_format=cfg.DATA_FORMAT,
        per_patch=cfg.EVAL_PER_PATCH,
        random_sample_size=cfg.RANDOM_SAMPLE_SIZE,
        path_to_color_map=os.path.join(cfg.DATA_ROOT, 'sem_color_labels.txt'),
        eval_format=cfg.EVAL_FORMAT,
        mean_type=cfg.MEAN_TYPE,
        drop_unknown=cfg.DROP_UNKNOWN,
        norm_stats=norm_stats,
    )

    # ================================
    # Run training or evaluation
    # ================================
    if args.evaluate:
        engine.evaluate(checkpoint_path)
    else:
        engine.train(checkpoint_path, load_weights_only)
