import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
import segmentation_models_pytorch as smp
import os


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

# Testing
# if __name__ == "__main__":
#     # bottleneck = torch.randn(1, 64, 80, 80)  # Bottleneck output (x3)
#     # skip1 = torch.randn(1, 32, 160, 160)    # Skip connection 1 (x2)
#     # skip2 = torch.randn(1, 16, 320, 320)    # Skip connection 2 (x1)
#     # skip3 = torch.randn(1, 8, 640, 640)     # Skip connection 3 (x)

#     # # Instantiate the decoder
#     # decoder = UNetDecoder(n_classes=1)

#     # # Forward pass
#     # output = decoder(bottleneck, [skip1, skip2, skip3])
#     # print(output.shape)  # Expected output: (1, 1, 640, 640)
    
#     input_tensor = torch.randn(1, 1, 640, 640)
#     model_dir = os.path.join('..', 'model')
#     encoder_path = os.path.join(model_dir, 'autoencoder.pth')

#     autoencoder_encoder = conv_autoencoder()
#     autoencoder_encoder.load_state_dict(torch.load(encoder_path))
#     unet_decoder = UNetDecoder(n_classes=1)
    
#     # Create HybridUNet model
#     model = HybridUNet(autoencoder_encoder, unet_decoder)

#     # Test the model
#     output = model(input_tensor)
#     print(output.shape)  # Should print torch.Size([1, 1, 640, 640])