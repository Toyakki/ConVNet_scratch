import torch
import torch.nn as nn

class ViTSegmentation(nn.Module):
    def __init__(self, vit_model, num_classes):
        super(ViTSegmentation, self).__init__()
        self.vit = vit_model
        self.decoder =  nn.Sequential(
            nn.ConvTranspose2d(1000, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        features = self.vit(x)
        features = features.unsqueeze(2).unsqueeze(3)
        features = features.expand(-1, 1000, 14, 14)
        output = self.decoder(features)
        output = nn.functional.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)
        return output
