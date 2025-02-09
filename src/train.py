import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from models import get_unet_model
from dataset import SegmentedMarsDataset
from utils import dice_coefficient, plot_losses

# Configurations and Paths
model_dir = os.path.join('..', 'model')
best_model_path = os.path.join(model_dir, 'unet_best_model.pth')
best_dice_model_path = os.path.join(model_dir, 'unet_best_dice_model.pth')

source = os.path.join('..', 'data')
train_imgs_path = os.path.join(source, 'train/images')
train_masks_path = os.path.join(source, 'mask/train')
val_imgs_path = os.path.join(source, 'valid/images')
val_masks_path = os.path.join(source, 'mask/valid')

# Hyperparameters and Device
seed = 42
lr = 1e-4
num_epochs = 10
batch_size = 16
tolerance = 0.001
pin_enabled = False

device = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

# Set seeds for reproducibility
torch.manual_seed(seed)
if device == "mps":
    try:
        torch.mps.manual_seed(seed)
    except AttributeError:
        print("MPS manual_seed not available; proceeding without it.")

# Create the model, loss, optimizer, and scheduler
unet_model = get_unet_model(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(unet_model.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Define transforms
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
mask_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = SegmentedMarsDataset(train_imgs_path, train_masks_path,
                                       crop_size=(256, 256),
                                       transform=image_transform,
                                       mask_transform=mask_transform)
val_dataset = SegmentedMarsDataset(val_imgs_path, val_masks_path,
                                   crop_size=(256, 256),
                                   transform=image_transform,
                                   mask_transform=mask_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=pin_enabled)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=pin_enabled)

# Variables to track progress
best_val_loss = float('inf')
best_dice = 0.0
train_losses = []
val_losses = []

# Training Loop (without AMP)
for epoch in range(num_epochs):
    # Training Phase
    unet_model.train()
    epoch_loss = 0.0
    for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = unet_model(images)
        train_loss = criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()
        epoch_loss += train_loss.item()
    avg_train_loss = epoch_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

    # Validation Phase
    unet_model.eval()
    total_val_loss = 0.0
    total_val_dice = 0.0
    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = unet_model(images)
            loss = criterion(outputs, masks)
            total_val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).to(outputs.dtype)
            total_val_dice += dice_coefficient(preds, masks)
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_dice = total_val_dice / len(val_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}')
    
    # Record losses
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save best models (with a tolerance to prevent frequent overwrites)
    if avg_val_loss < best_val_loss - tolerance:
        best_val_loss = avg_val_loss
        torch.save(unet_model.state_dict(), best_model_path)
        print(f"Saved Best Model at Epoch {epoch + 1}")
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(unet_model.state_dict(), best_dice_model_path)
        print(f"Saved Best Dice Score Model at Epoch {epoch + 1}")
    
    # Update scheduler using validation loss
    scheduler.step(avg_val_loss)

plot_losses(train_losses, val_losses)
