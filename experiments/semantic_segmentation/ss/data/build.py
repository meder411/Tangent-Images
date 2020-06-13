import os
import torch
import tqdm

from spherical_distortion.util import get_rank, synchronize
from .synthia import OmniSynth
from .stanford import StanfordDataset

NORMALIZATIONS = {
    'ugscnn': ([0.4974898, 0.47918808, 0.42809588, 1.0961773],
               [0.23762763, 0.23354423, 0.23272438, 0.75536704]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'stanford-rgb':
    ([0.5490018535423888, 0.5305735878705738, 0.477942701493174],
     [0.19923060886633429, 0.20050344344849544, 0.21290783912717565]),
    'synthia': ([0.45120219, 0.42573797, 0.39205981],
                [0.26007294, 0.26803442, 0.29314858]),
}


def cache_dataset(dataset):
    print('Caching data to {}.....'.format(dataset.cache_folder))
    for i in tqdm.tqdm(range(len(dataset))):
        sample = dataset.__getitem__(i, tocache=True)


def build_dataset(cfg, is_train=True):

    # To expedite training, resize equirectangular inputs by up to a factor of 4
    scale_denom = min(4, 2**(10 - cfg.SAMPLE_ORDER))
    scale_factor = 1.0 / scale_denom

    if isinstance(cfg.NORMALIZATION_STATS, str):
        # normalization stats are provided as string
        mean, std = NORMALIZATIONS[cfg.NORMALIZATION_STATS]
    else:
        # normalization stats are provided explicitly
        mean, std = cfg.NORMALIZATION_STATS

    if cfg.DATASET == 'synthia':
        dataset = OmniSynth(
            omni_dp=cfg.DATA_ROOT,
            data_format=cfg.DATA_FORMAT,
            is_train=is_train,
            fov=(cfg.FOV, cfg.FOV),
            dim=(cfg.DIM, cfg.DIM),
            scale_factor=scale_factor,
            cache_root=cfg.CACHE_ROOT,
            mean=mean,
            std=std,
        )
    elif cfg.DATASET == 'stanford':
        fold = cfg.FOLD
        file_name = 'train.txt' if is_train else 'test.txt'
        file_list = os.path.join('stanford-data-lists', cfg.DATA_FORMAT,
                                 'fold{}'.format(fold), file_name)

        dataset = StanfordDataset(
            root_path=cfg.DATA_ROOT,
            path_to_img_list=file_list,
            normalize_intrinsics=cfg.NORMALIZE_INTRINSICS,
            fold=cfg.FOLD,
            fov=(cfg.FOV, cfg.FOV),
            dim=(cfg.DIM, cfg.DIM),
            scale_factor=scale_factor,
            data_format=cfg.DATA_FORMAT,
            use_depth=cfg.USE_DEPTH,
            cache_root=cfg.CACHE_ROOT,
            mean=mean,
            std=std,
        )
    else:
        raise AttributeError('Dataset {} is not implemented'.format(
            cfg.DATASET))

    return dataset


def build_dataloader(cfg, num_gpus, distributed=False, is_train=True):

    batch_size = cfg.BATCH_SIZE_PER_GPU
    batch_size = batch_size if distributed else batch_size * num_gpus
    num_workers = cfg.NUM_WORKERS_PER_GPU
    num_workers = num_workers if distributed else num_workers * num_gpus

    dataset = build_dataset(cfg, is_train)

    if cfg.CACHE:
        if distributed:
            # In distributed mode, only have one process caching
            if get_rank() == 0:
                cache_dataset(dataset)
            synchronize()
        else:
            cache_dataset(dataset)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True)
    elif is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    drop_last = is_train
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return dataloader
