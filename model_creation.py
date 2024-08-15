"""
Ryan Tietjen
Aug 2024
Contains helper functions related to creating various machine learning models
"""

import torchvision
import torch
from torch import nn

def create_linear_model():
    """
    TODO: Implement
    """
    pass

def CNN():
    pass

def ViT():
    pass


def effnetb2(num_classes:int=101, 
                          seed:int=42,
                          freeze_gradients:bool=True):
    
    torch.manual_seed(seed)
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights)

    if freeze_gradients:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    
    return model, transforms