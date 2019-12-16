import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


class Model(nn.Module):

    def __init__(self, nclasses, feat=32):
        super().__init__()
        self.in_conv = nn.Conv2d(6, feat, 3, stride=2, padding=1)
        self.in_bn = nn.BatchNorm2d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
        self.block1 = ResBlock(
            in_chan=feat,
            neck_chan=feat,
            out_chan=4 * feat,
            coarsen=True,
            scale_factor=0.5)
        self.block2 = ResBlock(
            in_chan=4 * feat,
            neck_chan=4 * feat,
            out_chan=16 * feat,
            coarsen=True,
            scale_factor=0.5)
        self.block3 = ResBlock(
            in_chan=16 * feat,
            neck_chan=16 * feat,
            out_chan=64 * feat,
            coarsen=True,
            scale_factor=0.5)
        self.out_layer = nn.Linear(64 * feat, nclasses)

    def forward(self, patches):
        patches_out = []
        for i in range(patches.shape[1]):
            x = patches[:, i, ...]
            x = self.in_block(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            B, C, _, _ = x.shape
            x = x.view(B, C, -1)
            patches_out.append(x)
        x = torch.cat(patches_out, -1)
        N = x.shape[-1]
        x = F.avg_pool1d(x, N).squeeze(-1)  # output: batch x channels x 1
        x = F.dropout(x, training=self.training)
        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)


class ResBlock(nn.Module):

    def __init__(self, in_chan, neck_chan, out_chan, coarsen, scale_factor=0.5):
        super().__init__()
        self.coarsen = coarsen
        self.conv1 = nn.Conv2d(in_chan, neck_chan, 1)
        self.conv2 = nn.Conv2d(neck_chan, neck_chan, 3, padding=1)
        self.conv3 = nn.Conv2d(neck_chan, out_chan, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(neck_chan)
        self.bn2 = nn.BatchNorm2d(neck_chan)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.down = Interpolate(scale_factor)
        self.diff_chan = (in_chan != out_chan)

        if coarsen:
            self.seq1 = nn.Sequential(self.conv1, self.down, self.bn1,
                                      self.relu, self.conv2, self.bn2,
                                      self.relu, self.conv3, self.bn3)
        else:
            self.seq1 = nn.Sequential(self.conv1, self.bn1, self.relu,
                                      self.conv2, self.bn2, self.relu,
                                      self.conv3, self.bn3)

        if self.diff_chan or coarsen:
            self.conv_ = nn.Conv2d(in_chan, out_chan, 1)
            self.bn_ = nn.BatchNorm2d(out_chan)
            if coarsen:
                self.seq2 = nn.Sequential(self.conv_, self.down, self.bn_)
            else:
                self.seq2 = nn.Sequential(self.conv_, self.bn_)

    def forward(self, x):
        if self.diff_chan or self.coarsen:
            x2 = self.seq2(x)
        else:
            x2 = x
        x1 = self.seq1(x)
        out = x1 + x2
        out = self.relu(out)
        return out


class Interpolate(nn.Module):

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)