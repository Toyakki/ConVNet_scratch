import numpy as np
import random

def sample_regions(image, bounding_boxes, crop_size, num_samples=5):
    img_height, img_width = image.shape
    cropped_regions = []  
    
    mask = np.zeros_like(image, dtype=np.uint8)
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        mask[y_min:y_max, x_min:x_max] = 1  # Mark crater regions on the mask

    for _ in range(num_samples):
        valid_crop = False
        attempts = 0
        while not valid_crop and attempts < 10:  # Limit attempts to find a valid region
            # Randomly sample a top-left corner
            x_start = random.randint(0, img_width - crop_size)
            y_start = random.randint(0, img_height - crop_size)
            x_end = x_start + crop_size
            y_end = y_start + crop_size
            
            # Ensure the crop is within bounds and doesn't overlap with craters
            if x_end <= img_width and y_end <= img_height:
                crop_mask = mask[y_start:y_end, x_start:x_end]
                if np.sum(crop_mask) == 0:  # Ensure no crater pixels in the crop
                    cropped_regions.append(image[y_start:y_end, x_start:x_end])
                    valid_crop = True
            attempts += 1

    return cropped_regions