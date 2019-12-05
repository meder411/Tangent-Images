import torch
import torchvision
import os.path as osp
from torch import nn

from .zhang_unet import ZhangUNet
from .hexunet import HexUNet
from .. import comm


def build_model(cfg, num_classes):
    comm.dprint('Initializing network')
    in_ch = 4 if cfg.USE_DEPTH else 3

    if cfg.MODEL_TYPE == 'zhangunet':
        model = ZhangUNet(in_ch=in_ch,
                          out_ch=num_classes,
                          input_nonlin=cfg.INPUT_NONLIN)
        train_mode = True
    elif cfg.MODEL_TYPE == 'resnet101':
        model = torchvision.models.segmentation.fcn_resnet101(
            pretrained=False,
            num_classes=num_classes
        )
        train_mode = False
    elif cfg.MODEL_TYPE == 'hexunet':
        model = HexUNet(num_classes)
        train_mode = True
    else:
        raise AttributeError(
            'Model type {} is not supported.'.format(cfg.MODEL_TYPE))

    if cfg.MODEL_TYPE == 'resnet101':
        url = 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
        state_dict = torch.hub.load_state_dict_from_url(url)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict
                      and v.shape == model_dict[k].shape}
        # print(state_dict.keys())
        model.load_state_dict(state_dict, strict=False)
        train_mode = False

        if cfg.USE_DEPTH:
            old_conv = model.backbone.conv1
            inplanes = old_conv.out_channels
            model.backbone.conv1 = nn.Conv2d(
                4, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.constant_(model.backbone.conv1.weight, 0.0)
            model.backbone.conv1.weight.data[:, :3] = old_conv.weight.detach()

    return model, train_mode


def get_checkpoint_path(checkpoint_dir, evaluate, start_epoch):
    if start_epoch == 0:
        checkpoint_path = None if not evaluate \
            else osp.join(checkpoint_dir, 'model_best.pth')
    elif start_epoch == -1:
        checkpoint_path = osp.join(
            checkpoint_dir, 'checkpoint_latest.pth')
    else:
        checkpoint_path = osp.join(
            checkpoint_dir, 'checkpoint_{:03d}.pth'.format(start_epoch))
    load_weights_only = evaluate

    return checkpoint_path, load_weights_only


LABEL_RATIOS = {
    'synthia-none': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.05],
    'synthia-ours': [0, 2.2332e-01, 2.5028e-01, 3.0387e-01, 3.7913e-02,
                     4.6966e-02, 4.3975e-02, 1.1833e-02, 5.0253e-02, 1.8071e-03,
                     1.7912e-03, 6.8025e-06, 2.7346e-02, 6.4137e-04],
    'stanford-ours': [
        0.0, 0.01574929912162261, 0.027120215697671146, 0.07173240823098485,
        0.09940005941682817, 0.03867709226962055, 0.11732465598112281,
        0.030507559126801635, 0.10977762878937447, 0.08451750469249816,
        0.003289480129178341, 0.02881729930964915, 0.3366731957269465,
        0.028805890997152264
    ],
    'stanford-thirdparty': [
        0.04233976974675504, 0.014504436907968913, 0.017173225930738712,
        0.048004778186652164, 0.17384037404789865, 0.028626771620973622,
        0.087541966989014, 0.019508096683310605, 0.08321331842901526,
        0.17002664771895903, 0.002515611224467519, 0.020731298851232174,
        0.2625963729249342, 0.016994731594287146, 0.012382599143792165][:-1],
}


def build_criterion(cfg):
    comm.dprint('Initializing training criteria')

    drop = [0]
    label_ratio = LABEL_RATIOS[cfg.LABEL_WEIGHT]
    label_ratio = torch.tensor(label_ratio)

    label_weight = torch.tensor(1.0) / torch.log(1.02 + label_ratio)
    label_weight[drop] = 0.
    label_weight = label_weight.to(dtype=torch.float32, device=cfg.DEVICE)

    if cfg.DROP_UNKNOWN:
        criterion = nn.CrossEntropyLoss(
            weight=label_weight[1:], ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=label_weight, ignore_index=0)
    criterion = criterion.to(cfg.DEVICE)
    return criterion


def build_optimizer(cfg, model):

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.LR,
    )
    return optimizer


def build_scheduler(cfg, optimizer):
    # TODO: avoid hard-coding parameters
    if cfg.SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 50],
            gamma=0.1)
    elif cfg.SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            200,
            gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.9
        )

    return scheduler
