# pylint: disable=E1101,R,C,W1202
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
from collections import OrderedDict
from time import time
import numpy as np

from tangent_images.util import *
from dataset import ModelNet, CacheNPY, ToMesh, ProjectOnSphere


def main(checkpoint_path, data_dir, dataset, partition, batch_size, feat,
         num_workers, image_shape, base_order, sample_order):

    torch.backends.cudnn.benchmark = True

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', "model.py")
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    num_classes = int(dataset[-2:])
    model = mod.Model(num_classes, feat=feat)
    model = nn.DataParallel(model)
    model = model.cuda()

    # load checkpoint
    ckpt = checkpoint_path
    pretrained_dict = torch.load(ckpt)
    load_partial_model(model, pretrained_dict)

    print("{} parameters in total".format(
        sum(x.numel() for x in model.parameters())))
    print("{} parameters in the last layer".format(
        sum(x.numel() for x in model.module.out_layer.parameters())))

    # Load the dataset
    # Increasing `repeat` will generate more cached files
    transform = CacheNPY(
        prefix='sp{}_'.format(sample_order),
        transform=torchvision.transforms.Compose([
            ToMesh(random_rotations=False, random_translation=0),
            ProjectOnSphere(
                dataset=dataset, image_shape=image_shape, normalize=True)
        ]))

    transform_test = CacheNPY(
        prefix='sp{}_'.format(sample_order),
        transform=torchvision.transforms.Compose([
            ToMesh(random_rotations=False, random_translation=0),
            ProjectOnSphere(
                dataset=dataset, image_shape=image_shape, normalize=True)
        ]))

    if dataset == 'modelnet10':

        def target_transform(x):
            classes = [
                'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
                'night_stand', 'sofa', 'table', 'toilet'
            ]
            return classes.index(x)
    elif dataset == 'modelnet40':

        def target_transform(x):
            classes = [
                'airplane', 'bowl', 'desk', 'keyboard', 'person', 'sofa',
                'tv_stand', 'bathtub', 'car', 'door', 'lamp', 'piano', 'stairs',
                'vase', 'bed', 'chair', 'dresser', 'laptop', 'plant', 'stool',
                'wardrobe', 'bench', 'cone', 'flower_pot', 'mantel', 'radio',
                'table', 'xbox', 'bookshelf', 'cup', 'glass_box', 'monitor',
                'range_hood', 'tent', 'bottle', 'curtain', 'guitar',
                'night_stand', 'sink', 'toilet'
            ]
            return classes.index(x)
    else:
        print('invalid dataset. must be modelnet10 or modelnet40')
        assert (0)

    test_set = ModelNet(
        data_dir,
        image_shape=image_shape,
        base_order=base_order,
        sample_order=sample_order,
        dataset=dataset,
        partition='test',
        transform=transform_test,
        target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)

    def test_step(data, target):
        model.eval()
        data, target = data.cuda(), target.cuda()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item()

    # test
    total_loss = 0
    total_correct = 0
    count = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        loss, correct = test_step(data, target)
        total_loss += loss
        total_correct += correct
        count += 1
        print("[Test] <LOSS>={:.2} <ACC>={:2}".format(
            total_loss / (count + 1), total_correct / len(test_set)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--checkpoint_path", type=str, required=True)

    # Dataset details
    parser.add_argument(
        "--partition", choices={"test", "train"}, default="test")
    parser.add_argument(
        "--dataset", choices={"modelnet10", "modelnet40"}, default="modelnet40")

    # Evaluation details
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--feat", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="data")

    # Tangent images details
    parser.add_argument('--image_shape', nargs='+', type=int, default=[64, 128])
    parser.add_argument('--base_order', type=int, default=0)
    parser.add_argument('--sample_order', type=int, default=5)

    args = parser.parse_args()

    main(**args.__dict__)