import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# Define a Double Convolution layer with optional dropout
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.conv1(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, backbone='resnet50', dropout_rate=0.0):
        super(Encoder, self).__init__()
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            # Reduced dropout in encoder blocks
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                *list(resnet.children())[1:3],
                nn.Dropout(dropout_rate),  # Dropout after the first block
            )
            self.enc2 = nn.Sequential(*list(resnet.children())[4])  # Removed dropout here
            self.enc3 = nn.Sequential(*list(resnet.children())[5])  # Removed dropout here
            self.enc4 = nn.Sequential(*list(resnet.children())[6])  # Removed dropout here
            self.enc5 = nn.Sequential(*list(resnet.children())[7])  # Removed dropout here
        else:
            raise ValueError("Only ResNet50 backbone is supported at the moment.")

    def forward(self, x):
        skip_connections = []
        x = self.enc1(x)
        skip_connections.append(x)
        x = self.enc2(x)
        skip_connections.append(x)
        x = self.enc3(x)
        skip_connections.append(x)
        x = self.enc4(x)
        skip_connections.append(x)
        x = self.enc5(x)
        return x, skip_connections

class Decoder(nn.Module):
    def __init__(self, features=[64, 256, 512, 1024], dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
        ])
        # Retained dropout in decoder but reduced dropout rate for experiments
        self.doubleconv = nn.ModuleList([
            DoubleConv(1024 + 1024, 1024, dropout_rate),  # Dropout here
            DoubleConv(512 + 512, 512, dropout_rate),    # Dropout here
            DoubleConv(256 + 256, 256, dropout_rate),    # Dropout here
            DoubleConv(64 + 64, 64, dropout_rate),      # Dropout here
        ])

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.up_convs)):
            x = self.up_convs[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.doubleconv[idx](x)
        return x

class UNet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 256, 512, 1024], backbone='resnet50', dropout_rate=0.0):
        super(UNet50, self).__init__()
        self.encoder = Encoder(in_channels, backbone=backbone, dropout_rate=dropout_rate)
        self.bottleneck = DoubleConv(2048, 2048, dropout_rate=dropout_rate)
        self.decoder = Decoder(features, dropout_rate=dropout_rate)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = F.interpolate(x, size=(99, 99), mode='bilinear', align_corners=False)
        return self.final_conv(x)
