import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

# Use Cases
# source = os.path.join('..', 'data')
# train_imgs_path = os.path.join(source, 'train/images')
# train_lbls_path = os.path.join(source, 'train/labels')
# train_segmented_path = os.path.join(source, 'processed/train')
# train_masks_path = os.path.join(source, 'mask/train')

# test_imgs_path = os.path.join(source, 'test/images')
# test_lbls_path = os.path.join(source, 'test/labels')
# test_segmented_path = os.path.join(source, 'processed/test')
# test_masks_path = os.path.join(source, 'mask/test')

# val_imgs_path = os.path.join(source, 'valid/images')
# val_lbls_path = os.path.join(source, 'valid/labels')
# val_segmented_path = os.path.join(source, 'processed/valid')
# val_masks_path = os.path.join(source, 'mask/valid')

# print(len(os.listdir(train_imgs_path)))
# print(len(os.listdir(train_masks_path)))

# dataset = SegmentedMarsDataset(train_imgs_path, train_masks_path, crop_size=(256, 256))

# # Create DataLoader
# batch_size = 8
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Check the DataLoader
# for batch in dataloader:
#     input_images, segmented_masks = batch

#     print("Input Images Shape:", input_images[0].shape)
#     print("Segmented Masks Shape:", segmented_masks[0].shape)
    
#     image, mask = input_images[0].squeeze().cpu().numpy(), segmented_masks[0].squeeze().cpu().numpy()
#     print(image.shape)
    
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     axes[0].imshow(image, cmap='gray')
#     axes[0].set_title("First image")
#     axes[0].axis('off')

#     axes[1].imshow(mask, cmap='gray')
#     axes[1].set_title("First mask")
#     axes[1].axis('off')

#     plt.show()
#     break
