import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from transformer.SegViT import ViTSegmentation
from transformer.MViT import MViTv2SegmentationModel
from torchvision.models import vit_b_16, ViT_B_16_Weights
from utils import dice_coefficient, plot_losses
from transformer_dataset import SegmentedMarsDataset

# Configs
source = os.path.join('..', 'data')
train_imgs_path = os.path.join(source, 'train/images')
train_lbls_path = os.path.join(source, 'train/labels')
train_segmented_path = os.path.join(source, 'processed/train')
train_masks_path = os.path.join(source, 'mask/train')

val_imgs_path = os.path.join(source, 'valid/images')
val_lbls_path = os.path.join(source, 'valid/labels')
val_segmented_path = os.path.join(source, 'processed/valid')
val_masks_path = os.path.join(source, 'mask/valid')

model_dir = os.path.join('..', 'model')
best_model_path = os.path.join(model_dir, 'vit_best_model.pth')
best_dice_model_path = os.path.join(model_dir, 'vit_best_dice_model.pth')

visualise = False
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
num_epochs = 10
# Visualise function

train_losses = []
val_losses = []
def check_loader(dataloader, visualise):
        # Visualize the first image and mask
    images, masks = next(iter(dataloader))
    print("The input shape of the image", images[0].shape)
    print("The input mask of the image", masks[0].shape)

    if visualise:
        image = images[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC for visualization
        mask = masks[0].squeeze().cpu().numpy()          # Squeeze mask to 2D
        # Denormalize the image for visualization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image * std) + mean
        image = (image * 255).astype('uint8')

        # Plot the image and mask
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Segmented Mask")
        axes[1].axis('off')

        plt.show()

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 for pretrained model compatibility
    transforms.ToTensor(),         # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet models
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match image size
    transforms.ToTensor()           # Convert to tensor
])

train_dataset = SegmentedMarsDataset(
    images_path=train_imgs_path,
    masks_path=train_masks_path,
    transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = SegmentedMarsDataset(
    images_path=val_imgs_path,
    masks_path=val_masks_path,
    transform= image_transform,
    mask_transform=mask_transform
)

# Initialize dataloaders
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# model = MViTv2SegmentationModel().to(device)

weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights='DEFAULT')
segmentation_model = ViTSegmentation(vit_model=model, num_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()  # Suitable for binary segmentation
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop

best_val_loss = float('inf')
best_dice = 0.0

for epoch in range(num_epochs):
    segmentation_model.train()
    epoch_loss = 0.0
    for images, masks in tqdm(train_dataloader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        output = segmentation_model(images)
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader):.4f}")
    
    segmentation_model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_dataloader):
            images, masks = images.to(device), masks.to(device)
            output = segmentation_model(images)
            loss = criterion(output, masks)
            val_loss += loss
            preds = (torch.sigmoid(output) > 0.5).int()
            val_dice += dice_coefficient(preds, masks)
    
    val_loss /= len(val_dataloader)
    val_dice /= len(val_dataloader)
    print(f'Epoch [{epoch + 1}/ {num_epochs}], Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}')
    # Add this to the end of each epoch
    train_losses.append(epoch_loss / len(train_dataloader))
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(segmentation_model.state_dict(), best_model_path)
        print(f"Saved Best Model at Epoch {epoch + 1}")
        
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(segmentation_model.state_dict(), best_dice_model_path)
        print(f"Saved Best dice score model at Epoch {epoch + 1}")

plot_losses(train_losses, val_losses)