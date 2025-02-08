import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import dice_coefficient


from models import get_unet_model, conv_autoencoder, UNetDecoder, HybridUNet
from dataset import SegmentedMarsDataset
from utils import dice_coefficient

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
batch_size = 8 
visualize = True

source = os.path.join('..', 'data')
test_imgs_path = os.path.join(source, 'test/images')
test_segmented_path = os.path.join(source, 'processed/test')
test_masks_path = os.path.join(source, 'mask/test')

model_dir = os.path.join('..', 'model')
encoder_path = os.path.join(model_dir, 'autoencoder.pth')
best_model_path = os.path.join(model_dir, 'unet_best_model.pth')
best_dice_model_path = os.path.join(model_dir, 'unet_best_dice_model.pth')


def run_inference(unet_model, test_loader, device, threshold=0.5, visualize=False):
    unet_model.eval()  # Set the unet_model to evaluation mode
    dice_scores = []

    with torch.no_grad():  # Disable gradient computation for inference
        for (images, masks) in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = unet_model(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Threshold probabilities to get binary predictions
            preds = (probs > threshold).float()

            # Compute Dice coefficient for each image
            for pred, mask in zip(preds, masks):
                dice = dice_coefficient(pred, mask)
                dice_scores.append(dice.item())

            if visualize:
                visualize_pred(images, masks, preds)

    # Compute average Dice score
    avg_dice = sum(dice_scores) / len(dice_scores)
    return avg_dice

def visualize_pred(images, masks, preds):
    for i in range(images.size(0)):  # Iterate through the batch
        img = images[i].squeeze().cpu().numpy()
        mask = masks[i].squeeze().cpu().numpy()
        pred = preds[i].squeeze().cpu().numpy()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred, cmap="gray")

        plt.show()


# Inference loop
test_dataset = SegmentedMarsDataset(test_imgs_path, test_masks_path, crop_size=(256, 256))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# unet_model = get_unet_model(num_classes=1).to(device)
autoencoder = conv_autoencoder()
autoencoder.load_state_dict(torch.load(encoder_path))
decoder = UNetDecoder(n_classes=1)
unet_model = HybridUNet(autoencoder, decoder).to(device)
# unet_model.load_state_dict(torch.load(best_model_path))
unet_model.load_state_dict(torch.load(best_dice_model_path))

unet_model.to(device)

# Run inference
avg_dice = run_inference(unet_model, test_dataloader, device, threshold=0.5, visualize=visualize)
print(f"Average Dice Coefficient: {avg_dice:.4f}")