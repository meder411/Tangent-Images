import argparse
import numpy as np
import random
import os
import torch
import visdom

from ss import comm
from ss.config_defaults import _C as cfg
from ss.data import build_dataloader
from ss.models import build_model, get_checkpoint_path
from ss.models import build_optimizer, build_criterion, build_scheduler
from ss.training_manager import PerspectiveManagerPanoSemSeg
from ss.testing_manager import TextureBakedTestingManagerSemSeg


def parse_args():
    parser = argparse.ArgumentParser(description='Tangent planes in PyTorch')
    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        help='Path to configuration file.',
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Launch in distributed mode. Please use run_dist_train tool.'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate the trained model.'
    )
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
    assert not (args.distributed and args.evaluate), \
        'Distributed eval is not supported.'


if __name__ == '__main__':
    # Parse command line arguments.
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.overwrite_args)
    cfg.freeze()
    check_arguments(cfg, args)
    comm.dprint('Running with config:')
    comm.dprint(cfg.dump())

    # Set random seeds for repeatability
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Speed up cuDNN
    torch.backends.cudnn.benchmark = True

    # Initialize devices and runtime mode.
    comm.initialize(args.local_rank, args.distributed)
    device_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(device_ids)
    comm.dprint('Using devices {}'.format(device_ids))

    # Experiment name and folders
    suffix = '' if not len(cfg.EXPERIMENT_SUFFIX) else '-' + args.suffix
    experiment_name = '{}-{}-patches-{}-l{}b{}-fold{}-bs{}{}'.format(
        cfg.DATASET,
        cfg.DATA_FORMAT,
        cfg.MODEL_TYPE,
        cfg.SAMPLE_ORDER,
        cfg.BASE_ORDER,
        cfg.FOLD,
        args.BATCH_SIZE_PER_GPU,
        suffix
    ) if not len(cfg.EXPERIMENT_NAME) else cfg.EXPERIMENT_NAME
    checkpoint_dir = os.path.join('experiments', experiment_name)
    checkpoint_path, load_weights_only = get_checkpoint_path(
        checkpoint_dir, args.evaluate, cfg.START_EPOCH)
    os.makedirs(checkpoint_dir, exist_ok=True)
    path_to_color_map = os.path.join(cfg.DATA_ROOT, 'sem_color_labels.txt')

    # Visdom visualization
    vis = visdom.Visdom(server=cfg.VISDOM.SERVER, env=experiment_name) \
        if args.local_rank == 0 else None

    # Datasets
    train_dataloader = build_dataloader(cfg, num_gpus, args.distributed,
                                        is_train=True)
    val_dataloader = build_dataloader(cfg, num_gpus, args.distributed,
                                      is_train=False)
    num_classes = train_dataloader.dataset.num_classes

    # Model
    model, train_mode = build_model(cfg, num_classes)
    if not args.distributed:
        model = torch.nn.DataParallel(
            model, device_ids=device_ids)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    model = model.to(cfg.DEVICE)

    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)
    scheduler = build_scheduler(cfg, optimizer)

    # TODO: avoid hard-coding parameters here
    H, W = (2048, 4096)
    scale_denom = min(8, 2 ** (10 - cfg.SAMPLE_ORDER))
    comm.dprint('Scale denom = {}'.format(scale_denom))
    image_shape = (H // scale_denom, W // scale_denom)

    if args.evaluate:
        sample_dir = os.path.join('samples', experiment_name)
        os.makedirs(sample_dir, exist_ok=True)
        tester = TextureBakedTestingManagerSemSeg(
            sample_dir=sample_dir,
            base_order=cfg.BASE_ORDER,
            max_sample_order=cfg.SAMPLE_ORDER,
            image_shape=image_shape,
            network=model,
            checkpoint_dir=checkpoint_dir,
            dataloader=val_dataloader,
            path_to_color_map=path_to_color_map,
            evaluation_sample_freq=cfg.SAMPLE_FREQ,
            device=cfg.DEVICE,
            drop_unknown=args.drop_unknown,
        )

        tester.evaluate(checkpoint_path)
    else:
        comm.dprint('Initializing training manager')
        batch_size = train_dataloader.batch_size
        effective_batch_size = batch_size
        stats = (train_dataloader.dataset.mean, train_dataloader.dataset.std)
        visualization_freq = 15
        validation_freq = 1
        trainer = PerspectiveManagerPanoSemSeg(
            network=model,
            checkpoint_dir=checkpoint_dir,
            name=experiment_name,
            base_order=cfg.BASE_ORDER,
            max_sample_order=cfg.SAMPLE_ORDER,
            image_shape=image_shape,
            random_sample_size=cfg.RANDOM_SAMPLE_SIZE,
            batch_size=batch_size,
            effective_batch_size=effective_batch_size,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=val_dataloader,
            path_to_color_map=path_to_color_map,
            criterion=criterion,
            optimizer=optimizer,
            visdom=vis,
            scheduler=scheduler,
            num_epochs=cfg.NUM_EPOCHS,
            validation_freq=validation_freq,
            visualization_freq=visualization_freq,
            evaluation_sample_freq=cfg.SAMPLE_FREQ,
            device=cfg.DEVICE,
            drop_unknown=cfg.DROP_UNKNOWN,
            stats=stats,
            distributed=args.distributed,
            local_rank=args.local_rank,
            train_mode=train_mode,
            data_format=cfg.DATA_FORMAT,
        )

        trainer.train(checkpoint_path, load_weights_only)
