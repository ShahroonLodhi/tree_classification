import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Your CNN model (simplified for example)
class CNN(nn.Module):
  def __init__(self, num_classes):
    super(CNN,self).__init__()

    self.features = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 512, 4, 2, 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
    )


    flattened_size = 512 * 8 * 8 

    self.Classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flattened_size, 256), 
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes) 
    )

  def forward(self, x):
    x = self.features(x)
    x = self.Classifier(x)
    return x

# Load pretrained U-Net

class UNet(smp.Unet):
    def __init__(self):
        super().__init__(encoder_name="resnet34", in_channels=3, classes=1)