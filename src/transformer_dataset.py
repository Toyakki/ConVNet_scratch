from torch.utils.data import Dataset
from PIL import Image
import os

class SegmentedMarsDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, mask_transform=None):
        self.images_path = sorted([os.path.join(images_path, f) for f in os.listdir(images_path)])
        self.masks_path = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path)])
        self.transform = transform
        self.mask_transform = mask_transform
    
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx]).convert("RGB")
        mask = Image.open(self.masks_path[idx]).convert("L")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask