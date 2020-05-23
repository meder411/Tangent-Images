import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_chan, neck_chan, out_chan, coarsen):
        super().__init__()
        self.coarsen = coarsen
        self.conv1 = nn.Conv2d(in_chan, neck_chan, 1)
        self.conv2 = nn.Conv2d(neck_chan, neck_chan, 3, padding=1)
        self.conv3 = nn.Conv2d(neck_chan, out_chan, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(neck_chan)
        self.bn2 = nn.BatchNorm2d(neck_chan)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.down = Interpolate(0.5)
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
            align_corners=False,
            recompute_scale_factor=True)


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        """
        """
        super().__init__()
        half_in = int(in_ch / 2)
        self.up = nn.ConvTranspose2d(
            half_in, half_in, kernel_size=4, stride=2, padding=1)
        self.conv = ResBlock(in_ch, out_ch, out_ch, False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        """
        """
        super().__init__()
        self.conv = ResBlock(in_ch, in_ch, out_ch, True)

    def forward(self, x):
        x = self.conv(x)
        return x


class ZhangUNet(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 num_blocks=5,
                 fdim=16,
                 input_nonlin=False):
        super().__init__()
        self.fdim = fdim
        self.num_blocks = num_blocks
        self.down = []
        self.up = []
        self.in_conv = nn.Conv2d(in_ch, fdim, 3, padding=1)
        self.input_nonlin = input_nonlin
        self.out_conv = nn.Conv2d(fdim, out_ch, 3, padding=1)
        # Downward path
        for i in range(self.num_blocks - 1):
            self.down.append(Down(fdim * (2**i), fdim * (2**(i + 1))))
        self.down.append(
            Down(fdim * (2**(self.num_blocks - 1)),
                 fdim * (2**(self.num_blocks - 1))))
        # Upward path
        for i in range(self.num_blocks - 1):
            self.up.append(
                Up(fdim * (2**(self.num_blocks - i)),
                   fdim * (2**(self.num_blocks - i - 2))))
        self.up.append(Up(fdim * 2, fdim))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x_ = [self.in_conv(x)]
        if self.input_nonlin:
            x_[0] = F.relu(x_[0], inplace=True)
        for i in range(self.num_blocks):
            x_.append(self.down[i](x_[-1]))
        x = self.up[0](x_[-1], x_[-2])
        for i in range(self.num_blocks - 1):
            x = self.up[i + 1](x, x_[-3 - i])
        x = self.out_conv(x)
        return x
