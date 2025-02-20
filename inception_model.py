import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, backbone='inception_v3', dropout_rate=0.0):
        super(Encoder, self).__init__()
        inception = models.inception_v3(weights='IMAGENET1K_V1')
        inception.aux_logits = False
        
        original_conv = inception.Conv2d_1a_3x3.conv
        inception.Conv2d_1a_3x3.conv = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        if in_channels == 1:
            inception.Conv2d_1a_3x3.conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        self.enc1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.maxpool1,
            nn.Dropout(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.maxpool2,
            nn.Dropout(dropout_rate)
        )
        self.enc3 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            nn.Dropout(dropout_rate)
        )
        self.enc4 = nn.Sequential(
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            nn.Dropout(dropout_rate)
        )
        self.enc5 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        skips = []
        x = self.enc1(x); skips.append(x)
        x = self.enc2(x); skips.append(x)
        x = self.enc3(x); skips.append(x)
        x = self.enc4(x); skips.append(x)
        x = self.enc5(x)
        return x, skips

class Decoder(nn.Module):
    def __init__(self, features=[768, 288, 192, 64], dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(2048, features[0], 2, 2),
            nn.ConvTranspose2d(features[0], features[1], 2, 2),
            nn.ConvTranspose2d(features[1], features[2], 2, 2),
            nn.ConvTranspose2d(features[2], features[3], 2, 2),
        ])
        
        self.dconvs = nn.ModuleList([
            DoubleConv(features[0]*2, features[0], dropout_rate),
            DoubleConv(features[1]*2, features[1], dropout_rate),
            DoubleConv(features[2]*2, features[2], dropout_rate),
            DoubleConv(features[3]*2, features[3], dropout_rate),
        ])

    def forward(self, x, skips):
        skips = skips[::-1]
        for i, (upconv, dconv) in enumerate(zip(self.upconvs, self.dconvs)):
            x = upconv(x)
            if x.shape[-2:] != skips[i].shape[-2:]:
                x = F.interpolate(x, size=skips[i].shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skips[i]], dim=1)
            x = dconv(x)
        return x

class UNetInception(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, dropout_rate=0.0):
        super(UNetInception, self).__init__()
        self.encoder = Encoder(in_channels, dropout_rate=dropout_rate)
        self.bottleneck = DoubleConv(2048, 2048, dropout_rate)
        self.decoder = Decoder(dropout_rate=dropout_rate)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        x = F.interpolate(x, size=(99, 99), mode='bilinear', align_corners=False)
        return self.final_conv(x) 