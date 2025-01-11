import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import get_unet_model, conv_autoencoder, UNetDecoder, HybridUNet
from dataset import SegmentedMarsDataset
from utils import dice_coefficient, plot_losses
# Configs
model_dir = os.path.join('..', 'model')
encoder_path = os.path.join(model_dir, 'autoencoder.pth')
best_model_path = os.path.join(model_dir, 'unet_best_model.pth')
best_dice_model_path = os.path.join(model_dir, 'unet_best_dice_model.pth')

source = os.path.join('..', 'data')
train_imgs_path = os.path.join(source, 'train/images')
train_segmented_path = os.path.join(source, 'processed/train')
train_masks_path = os.path.join(source, 'mask/train')

val_imgs_path = os.path.join(source, 'valid/images')
val_segmented_path = os.path.join(source, 'processed/valid')
val_masks_path = os.path.join(source, 'mask/valid')

# Hyperparameters
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
lr = 1e-5
num_epochs = 10
batch_size = 8


autoencoder = conv_autoencoder()
autoencoder.load_state_dict(torch.load(encoder_path))
decoder = UNetDecoder(n_classes=1)
unet_model = HybridUNet(autoencoder, decoder).to(device)

# unet_model = get_unet_model(num_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(unet_model.parameters(), lr)

# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Create dataset and dataloader
train_dataset = SegmentedMarsDataset(train_imgs_path, train_masks_path, crop_size=(256, 256))
val_dataset =  SegmentedMarsDataset(val_imgs_path, val_masks_path, crop_size=(256, 256))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


best_val_loss = float('inf')
best_dice = 0.0
train_losses = []
val_losses = []


for epoch in range(num_epochs):
    unet_model.train()
    epoch_loss = 0.0
    for (images, targets) in tqdm(train_dataloader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = unet_model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}, Training Loss:{loss.item()}')

    
    unet_model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for image, masks in val_dataloader:
            image, masks = image.to(device), masks.to(device)
            outputs = unet_model(image)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            val_dice += dice_coefficient(preds, masks)
    
    val_loss /= len(val_dataloader)
    val_dice /= len(val_dataloader)
    print(f'Epoch [{epoch + 1}/ {num_epochs}], Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}')
    # Add this to the end of each epoch
    train_losses.append(epoch_loss / len(train_dataloader))
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(unet_model.state_dict(), best_model_path)
        print(f"Saved Best Model at Epoch {epoch + 1}")
        
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(unet_model.state_dict(), best_dice_model_path)
        print(f"Saved Best dice score model at Epoch {epoch + 1}")

    # scheduler.step(val_loss)

plot_losses(train_losses, val_losses)