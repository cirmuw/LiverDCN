import torch
import torch.nn as nn
import numpy as np


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class autoencoder(nn.Module):
    def __init__(self, inchannels=1, num_bottleneck=10):
        super(autoencoder, self).__init__()
        print(inchannels)
        self.encoder = nn.Sequential(
            nn.Conv2d(inchannels, 50, 3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2),  # b, 16, 5, 5
            nn.Conv2d(50, 20, 3, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 10, 3, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2)  # b, 8, 2, 2
        )
        
        self.bottleneck = nn.Sequential(nn.Linear(160, num_bottleneck), nn.ReLU(True))
        
        self.debottleneck = nn.Sequential(nn.Linear(num_bottleneck, 160), nn.ReLU(True))
        
        self.decoder = nn.Sequential(
            nn.Sequential(Interpolate(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(10, 20, kernel_size=1)),
            nn.ReLU(True),
            nn.Sequential(Interpolate(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(20, 50, kernel_size=1)),
            nn.ReLU(True),
            nn.Sequential(Interpolate(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(50, inchannels, kernel_size=1)),
            nn.ReLU(True),
            nn.Tanh()
        )
        
    def get_latent(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.bottleneck(x)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.debottleneck(x)
        x = x.view(x.size(0), 10, 4, 4)
        x = self.decoder(x)
        return x