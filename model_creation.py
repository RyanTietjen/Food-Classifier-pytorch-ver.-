"""
Ryan Tietjen
Aug 2024
Contains helper functions related to creating various machine learning models
"""

import torchvision
import torch
from torch import nn

def create_model(config,
                 num_classes:int=101, 
                 seed:int=31, 
                 freeze_gradients:bool=True):
    if config["Model Utilization"]["model_type"] == "effnetb2":
        return effnetb2(num_classes=num_classes,
                        seed=seed,
                        freeze_gradients=freeze_gradients)
    else:
        raise Exception("Invalid config provided")

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
             seed:int=31, 
             freeze_gradients:bool=True):
    torch.manual_seed(seed)

    #Create model and extract weights/transforms
    weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    #Freeze gradients to avoid modifying the original model    
    if freeze_gradients:
        for param in model.parameters():
            param.requires_grad = False

    #modify classifier model to fit our 
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    
    return model, transforms