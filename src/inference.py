import torch

def run_inference(model, test_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)

