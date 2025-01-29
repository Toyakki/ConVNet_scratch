import torch.nn as nn
import timm

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
