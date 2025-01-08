import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from unet_parts import Up

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # b, 320, 320
            nn.ReLU(True))
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True)
            )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 80, 80
            nn.ReLU(True)
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 160, 160
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 320, 320
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # 640, 640
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x = self.decoder(x3)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(UNetDecoder, self).__init__()
        self.bilinear = bilinear

        # Upsampling layers
        self.up1 = Up(64, 32, bilinear)  # From 64 channels to 32, upsample 80x80 -> 160x160
        self.up2 = Up(32, 16, bilinear)  # From 32 channels to 16, upsample 160x160 -> 320x320
        self.up3 = Up(16, 8, bilinear)   # From 16 channels to 8, upsample 320x320 -> 640x640

        # Final output layer
        self.outc = nn.Conv2d(8, n_classes, kernel_size=1)  # Final 1x1 convolution for output masks

    def forward(self, x, skips):
        """
        Forward pass of the decoder.
        :param x: Bottleneck input of shape (b, 64, 80, 80).
        :param skips: List of skip connection tensors [(b, 32, 160, 160), (b, 16, 320, 320)].
        """
        x = self.up1(x, skips[0])  # Up to (b, 32, 160, 160), add skip from 160x160
        x = self.up2(x, skips[1])  # Up to (b, 16, 320, 320), add skip from 320x320
        x = self.up3(x, skips[2])  # Up to (b, 8, 640, 640), add skip from 640x640
        logits = self.outc(x)      # Final output layer
        return logits


######################
##### HybridUNet class from implementation###
class HybridUNet(nn.Module):
    def __init__(self, autoencoder_encoder, unet_decoder):
        super(HybridUNet, self).__init__()
        
        # Encoder from Conv Autoencoder
        self.encoder = autoencoder_encoder

        # Decoder from U-Net
        self.up1 = unet_decoder.up1
        self.up2 = unet_decoder.up2
        self.up3 = unet_decoder.up3
        self.outc = unet_decoder.outc

    def forward(self, x):
        # Encoder
        x1, x2, x3 = self.encoder(x)  # Intermediate features from Conv Autoencoder

        # Decoder
        x = self.up1(x3, x2)  # Skip connection with x2
        x = self.up2(x, x1)   # Skip connection with x1
        logits = self.outc(x) # Final output
        return logits

# Pretrained unet
def get_unet_model(encoder_name="efficientnet-b7",
                   encoder_weights="imagenet",
                   in_channels=1,
                   num_classes=3):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    return model
