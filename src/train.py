import os
import torch
import torch.nn as nn

from utils import load_labels
from models import get_unet_model
from dataset import SegmentedMarsDataset
from torch.utils.data import DataLoader


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
lr = 1e-3
epochs = 10

unet_model = get_unet_model(num_classes=1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(unet_model.parameters(), lr)


def train_one_epoch(model, dataloader, optimizer, device):
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        
# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

# Directories
source = os.path.join('..', 'data')
train_imgs_path = os.path.join(source, 'train/images')
train_lbls_path = os.path.join(source, 'train/labels')
train_segmented_path = os.path.join(source, 'processed/train')

test_imgs_path = os.path.join(source, 'test/images')
test_lbls_path = os.path.join(source, 'test/labels')
test_segmented_path = os.path.join(source, 'processed/test')

val_imgs_path = os.path.join(source, 'valid/images')
val_lbls_path = os.path.join(source, 'valid/labels')
val_segmented_path = os.path.join(source, 'processed/valid')

train_labels, train_ids = load_labels(train_lbls_path)
val_labels, val_ids = load_labels(val_lbls_path)
test_labels, test_ids = load_labels(test_lbls_path)

# Create Dataset
dataset = SegmentedMarsDataset(train_imgs_path, train_segmented_path, crop_size=(256, 256))

# Create DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example: Iterate through the DataLoader
for batch in dataloader:
    input_images, segmented_masks = batch

    print("Input Images Shape:", input_images.shape)
    print("Segmented Masks Shape:", segmented_masks.shape)
    break
