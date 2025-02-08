# ConVNet_scratch
Exploring the modern Convnet architecture in the detection task. 

I am using the open sourced Moon crater dataset: https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset

The architecture I am planning to explore is:
1. AutoEncoder
2. UNet
3. YOLO
4. EfficientNet
5. Faster-RCNN
6. DETR (DEtection TRansformer)

Except for YOLO, I am trying to make some of their parts from scratch for better understanding.

## Notable results:

### UNet:
Not suprisingly, the pretrained model in smp library is giving the best results. I tried the hybrid Unet using autoencoder and UNet decoder from scratch, but the results were not great. 
The training loss and val loss change over 10 epochs. More traiings may result in good results.

![A sample result with a good crater result](Loss.png)

A sample result with 78% dice score.
![A sample result with a good crater result](Good_dice.png)

![A sample result with a bad crater result](Good_result.png)
