from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class CNNEncoder:
    """
    Extract image features using a pretrained ResNet-50.
    """
    def __init__(self, device: torch.device):
        self.device = device
        # Load ResNet-50 pretrained on ImageNet.
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(
            weights=weights
        )
        # Remove the final classification layer.
        #
        # Original:
        # ResNet-50 → 1000 ImageNet classes
        #
        # We want:
        # ResNet-50 → 2048 visual features
        self.model.fc = nn.Identity()   #pass the input as it is
        
        #Freeze Resnet
        for parameter in self.model.parameters():
            parameter.requires_grad = False
            
        self.model = self.model.to(self.device)    #move resnet to the device which we are using
        # Evaluation mode because we are using ResNet
        # only as a feature extractor.
        self.model.eval()
        # Use the preprocessing associated with the
        # pretrained ResNet-50 weights.
        self.transform = weights.transforms()       #resize,crop,convert to tensor,normalize,resnet50    
        
        
        