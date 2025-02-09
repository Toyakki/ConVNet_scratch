import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm

from unet_parts import Up


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),  # Retain spatial dimensions (640x640)
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # Downsample to 320x320
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Downsample to 160x160
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Downsample to 80x80
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Upsample to 160x160
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Upsample to 320x320
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # Upsample to 640x640
            nn.Tanh()
        )

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
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
        x1 = self.up1(x, skips[0])  # (b, 32, 160, 160)
        x2 = self.up2(x1, skips[1])  # (b, 16, 320, 320)
        x3 = self.up3(x2, skips[2])  # (b, 8, 640, 640)
        logits = self.outc(x3)
        return logits


######################
##### HybridUNet class from implementation###
class HybridUNet(nn.Module):
    def __init__(self, autoencoder_encoder, unet_decoder):
        super(HybridUNet, self).__init__()
        
        # Encoder from Conv Autoencoder
        self.encoder = autoencoder_encoder

        # Decoder from U-Net
        self.decoder = unet_decoder

    def forward(self, x):
        # Encoder
        x0 = self.encoder.encoder0(x)  # Feature map at 640x640
        x1 = self.encoder.encoder1(x0)  # Feature map at 320x320
        x2 = self.encoder.encoder2(x1)  # Feature map at 160x160
        x3 = self.encoder.encoder3(x2)  # Feature map at 80x80

        # Decoder
        logits = self.decoder(x3, [x2, x1, x0])  # Use x0 as the largest-scale skip connection
        return logits

# Use cases (For encoder-decoder architecture)
# autoencoder = conv_autoencoder()
# autoencoder.load_state_dict(torch.load(encoder_path))
# decoder = UNetDecoder(n_classes=1)
# unet_model = HybridUNet(autoencoder, decoder).to(device)

######################
## Pretrained models##

# Pretrained unet: smp.Unet
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

### MVIT
class MViTv2SegmentationModel(nn.Module):
    def __init__(self):
        super(MViTv2SegmentationModel, self).__init__()
        # Load timm pretrained MViTv2 model
        self.backbone = timm.create_model("mvitv2_tiny", pretrained=True, num_classes=0)
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)  # Get features from the backbone
        return self.segmentation_head(features)       # Pass features through segmentation head
