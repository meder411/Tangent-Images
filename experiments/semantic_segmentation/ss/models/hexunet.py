import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x,
                             scale_factor=self.scale_factor,
                             mode='bilinear',
                             align_corners=False)


class Encoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        """
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):

    def __init__(self, in_ch, neck_ch, out_ch):
        """
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, neck_ch, 3,
                                            padding=1), nn.BatchNorm2d(neck_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(neck_ch, neck_ch, 3, padding=1),
                                  nn.BatchNorm2d(neck_ch),
                                  nn.ReLU(inplace=True), Interpolate(2.0),
                                  nn.Conv2d(neck_ch, out_ch, 1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x1, x2=None):
        if x2 is None:
            return self.conv(x1)
        else:
            return self.conv(torch.cat((x1, x2), 1))


class HexUNet(nn.Module):

    def __init__(self, out_ch):
        super().__init__()
        self.enc0 = Encoder(3, 32)
        self.enc1 = Encoder(32, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.dec0 = Decoder(256, 512, 256)
        self.dec1 = Decoder(256 * 2, 256, 128)
        self.dec2 = Decoder(128 * 2, 128, 64)
        self.dec3 = Decoder(64 * 2, 64, 32)
        self.out0 = nn.Sequential(nn.Conv2d(32 * 2, 32, 3, padding=1),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x0_u = self.enc0(x)
        x0 = self.pool(x0_u)
        x1_u = self.enc1(x0)
        x1 = self.pool(x1_u)
        x2_u = self.enc2(x1)
        x2 = self.pool(x2_u)
        x3_u = self.enc3(x2)
        x3 = self.pool(x3_u)
        x4 = self.dec0(x3)
        x5 = self.dec1(x4, x3_u)
        x6 = self.dec2(x5, x2_u)
        x7 = self.dec3(x6, x1_u)
        x8 = self.out0(torch.cat((x7, x0_u), 1))
        x9 = self.out1(x8)
        x10 = self.out2(x9)
        return x10
