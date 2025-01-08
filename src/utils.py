import os
import pandas as pd
import torch

def load_labels(label_path):
    label_files = os.listdir(label_path)
    data = []
    image_ids = set()
    for file in label_files:
        image_name, _ = os.path.splitext(file)
        image = image_name + ".jpg"
        # Already checked every picture is in jpg format.
        image_ids.add(image)
        
        with open(os.path.join(label_path, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                box_info = list(map(float, line.strip().split()))
                data.append([image, *box_info])

    df = pd.DataFrame(data, columns=['file', 
                                     'class_id', 
                                     'center_x',
                                     'center_y',
                                     'width',
                                     'height'
                                    ])
    return df, image_ids

# Helper functions
def dice_coefficient(pred, target):
    smooth = 1.
    
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    
    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

# Use cases
# source = os.path.join('..', 'data')
# train_imgs_path = os.path.join(source, 'train/images')
# train_lbls_path = os.path.join(source, 'train/labels')

# test_imgs_path = os.path.join(source, 'test/images')
# test_lbls_path = os.path.join(source, 'test/labels')

# val_imgs_path = os.path.join(source, 'valid/images')
# val_lbls_path = os.path.join(source, 'valid/labels')

# train_labels, train_ids = load_labels(train_lbls_path)
# val_labels, val_ids = load_labels(val_lbls_path)
# test_labels, test_ids = load_labels(test_lbls_path)
