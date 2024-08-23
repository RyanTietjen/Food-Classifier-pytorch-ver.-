"""
Ryan Tietjen
Aug 2024
Creates a vit base 16 model for the demo
"""

import torch
import torchvision
from torch import nn

def vit_b_16(num_classes:int=101, 
             seed:int=31, 
             freeze_gradients:bool=True,
             unfreeze_blocks=0):
    """
    Initializes and configures a Vision Transformer (ViT-B/16) model with options for freezing gradients
    and adjusting the number of trainable blocks.

    This function sets up a ViT-B/16 model pre-trained on the ImageNet-1K dataset, modifies the classification
    head to accommodate a specified number of classes, and optionally freezes the gradients of certain blocks
    to prevent them from being updated during training.

    Parameters:
    num_classes (int): The number of output classes for the new classification head. Default is 101.
    seed (int): Random seed for reproducibility. Default is 31.
    freeze_gradients (bool): If True, freezes the gradients of the model's parameters, except for the last few
                             blocks specified by `unfreeze_blocks`. Default is True.
    unfreeze_blocks (int): Number of transformer blocks from the end whose parameters will have trainable gradients.
                           Default is 0, implying all are frozen except the new classification head.

    Returns:
    tuple: A tuple containing:
        - model (torch.nn.Module): The modified ViT-B/16 model with a new classifier head.
        - transforms (callable): The transformation function required for input images, as recommended by the
                                 pre-trained weights.

    Example:
    ```python
    model, transform = vit_b_16(num_classes=101, seed=31, freeze_gradients=True, unfreeze_blocks=2)
    ```

    Notes:
    - The total number of parameters in the model is calculated and used to determine which parameters to freeze.
    - The classifier head of the model is replaced with a new linear layer that outputs to the specified number of classes.
    """

    torch.manual_seed(seed) 

    #Create model and extract weights/transforms
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    params = list(model.parameters())
    params_to_unfreeze = 4 + (12 * unfreeze_blocks)
    # Total number of parameters
    total_params = len(params)


    #Freeze gradients to avoid modifying the original model    
    if freeze_gradients:
        for i, param in enumerate(params):
            # Set requires_grad to False for all but the last n encoder blocks
            if i < total_params - params_to_unfreeze:
                param.requires_grad = False

    #modify classifier model to fit our 
    model.heads = nn.Sequential(
        nn.Linear(in_features=768,
                  out_features=num_classes))
    
    return model, transforms