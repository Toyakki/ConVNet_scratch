import os
import cv2
import torch
from torch.utils.data import Dataset

class SegmentedMarsDataset(Dataset):
    def __init__(self, input_image_dir, segmented_mask_dir, crop_size=(256, 256)):
        """
        Dataset for training U-Net using segmented images from the autoencoder.
        """
        self.input_image_paths = sorted([os.path.join(input_image_dir, f) for f in os.listdir(input_image_dir)])
        self.segmented_mask_paths = sorted([os.path.join(segmented_mask_dir, f) for f in os.listdir(segmented_mask_dir)])
        self.crop_size = crop_size

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
        # Load input image
        input_image = cv2.imread(self.input_image_paths[idx], cv2.IMREAD_GRAYSCALE)
        input_image = cv2.resize(input_image, self.crop_size)
        input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize and add channel dim

        # Load segmented mask
        segmented_mask = cv2.imread(self.segmented_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        segmented_mask = cv2.resize(segmented_mask, self.crop_size)
        segmented_mask = torch.tensor(segmented_mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize and add channel dim

        return input_image, segmented_mask
