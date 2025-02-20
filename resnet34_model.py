import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv1(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, backbone='resnet34'):
        super(Encoder, self).__init__()
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:3]
        )
        self.enc2 = nn.Sequential(*list(resnet.children())[4])
        self.enc3 = nn.Sequential(*list(resnet.children())[5])
        self.enc4 = nn.Sequential(*list(resnet.children())[6])
        self.enc5 = nn.Sequential(*list(resnet.children())[7])

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
        skip_connections.append(x)
        return x, skip_connections

class Decoder(nn.Module):
    def __init__(self, features=[64, 128, 256, 512]):
        super(Decoder, self).__init__()
        self.up_convs = nn.ModuleList()
        self.doubleconv = nn.ModuleList()
        for feature in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.doubleconv.append(DoubleConv(feature * 2, feature))

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.up_convs)):
            x = self.up_convs[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.doubleconv[idx](x)
        return x

class UNet34(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, features=[64, 128, 256, 512]):
        super(UNet34, self).__init__()
        self.encoder = Encoder(in_channels)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.decoder = Decoder(features)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = F.interpolate(x, size=(99, 99), mode='bilinear', align_corners=False)
        return self.final_conv(x) 