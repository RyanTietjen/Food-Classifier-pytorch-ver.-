"""
Ryan Tietjen
Aug 2024
Contains helper functions related to creating various machine learning models
"""

import torchvision
from torchvision import transforms
import torch
from torch import nn
from pathlib import Path

def create_model(config, num_classes=101, seed=31, freeze_gradients=True, unfreeze_blocks=0):
    model_dispatch = {
        "linear": linear,
        "tiny_vgg": tiny_vgg,
        "effnetb2": effnetb2,
        "effnetb4": effnetb4,
        "vit_b_16": vit_b_16,
    }

    model_type = config["Model Utilization"]["model_type"]
    unfreeze_blocks = int(config["Model Utilization"]["num_blocks_to_unfreeze"])
    model_func = model_dispatch.get(model_type)

    if model_func:
        return model_func(num_classes=num_classes, seed=seed, freeze_gradients=freeze_gradients, unfreeze_blocks=unfreeze_blocks)
    else:
        raise Exception("Invalid config provided")

def linear(num_classes:int=101, 
             seed:int=31,
             freeze_gradients = False,
             unfreeze_blocks=0):
    """
    Initializes a simple linear model with a defined input size and a specific number of output classes,
    along with a transformation pipeline for input data preprocessing.

    This function creates a sequential model consisting of a flattening layer and two linear layers,
    designed to handle input images of a fixed size (224x224 pixels). It also sets up a series of 
    transformations to properly preprocess the input images for the model.

    Parameters:
    num_classes (int): The number of output classes for the final linear layer. Default is 101.
    seed (int): Random seed for reproducibility. Default is 31.
    freeze_gradients (bool): If True, freezes the gradients of the model's parameters. Even though this 
                             parameter exists, it currently does not influence the function as all parameters
                             are unfrozen by default. Default is False.
    unfreeze_blocks (int): Currently unused in the function. Reserved for potential future functionality 
                           where specific blocks of parameters can be unfrozen.

    Returns:
    tuple: A tuple containing:
        - model (torch.nn.Module): A sequential model with a series of linear transformations.
        - transform (callable): A transformation function for input image preprocessing, including resizing, 
                                normalization, and conversion to tensor format.

    Example:
    ```python
    model, transform = linear(num_classes=100, seed=42)
    ```

    Notes:
    - The function sets a random seed to ensure reproducibility of the initialization process.
    - The model expects input images of size 224x224 pixels, which are flattened into vectors of size 150528.
    - Normalization constants in the transform are standard values used with models pre-trained on the ImageNet dataset.
    - The `freeze_gradients` and `unfreeze_blocks` parameters are placeholders for potential future enhancements and do not
      currently affect the behavior of the function.
    """
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((224,224)),            
        transforms.ToTensor(),             
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])   
    ])
    model = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=150528, out_features=512), 
            nn.Linear(in_features=512, out_features=num_classes)
        )
    return model, transform


def tiny_vgg(num_classes:int=101, 
             seed:int=31,
             freeze_gradients = False,
             unfreeze_blocks=0): 
    """
    Initializes and returns a simplified VGG-like model with fewer and smaller convolutional layers, tailored
    to a specific number of classes, and a transformation pipeline for image preprocessing.
    Architecutre from https://poloclub.github.io/cnn-explainer/

    This function constructs a small-scale convolutional neural network reminiscent of the VGG architecture,
    but significantly reduced in complexity and size. It's designed to process images resized to 224x224 pixels.
    The model features several convolutional layers with ReLU activations and max pooling, followed by a
    flattening layer and a linear classifier.

    Parameters:
    num_classes (int): The number of output classes for the final linear layer. Default is 101.
    seed (int): Random seed for reproducibility across different runs. Default is 31.
    freeze_gradients (bool): If True, freezes the gradients of the model's parameters to prevent them from
                             being updated during training. Default is False.
    unfreeze_blocks (int): This parameter is currently unused but included for potential future extensions
                           where specific blocks of the model might be selectively unfrozen.

    Returns:
    tuple: A tuple containing:
        - model (torch.nn.Module): A tiny VGG-like sequential model configured for classification with
                                   the specified number of classes.
        - transform (callable): A transformation function for preprocessing input images, including resizing,
                                tensor conversion, and normalization.

    Example:
    ```python
    model, transform = tiny_vgg(num_classes=100, seed=42)
    ```

    Notes:
    - The transformation applied to input images includes resizing to 224x224 pixels, conversion to tensor,
      and normalization using mean and standard deviation values typical for models trained on the ImageNet dataset.
    - The model is designed to receive input tensors of shape (3, 224, 224), corresponding to 3-channel color images.
    - The `freeze_gradients` and `unfreeze_blocks` parameters are placeholders for future functionality and do not
      currently affect the function's behavior.
    """
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((224,224)),            
        transforms.ToTensor(),             
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])   
    ])
    hidden_units = 10
    model = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)),

            nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)),
            
            nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*3136,
                      out_features=num_classes)))
    return model, transform



def effnetb2(num_classes:int=101, 
             seed:int=31, 
             freeze_gradients:bool=True,
             unfreeze_blocks=0):
    """
    Initializes and customizes an EfficientNet-B2 model for a specific number of classes, with options
    to freeze model parameters to prevent updates during training.

    This function configures an EfficientNet-B2 model pre-trained on the ImageNet-1K dataset. It modifies
    the classifier head to handle a specified number of classes and offers the option to freeze the gradients
    of the model’s parameters entirely.

    Parameters:
    num_classes (int): The number of output classes for the new classifier head. Default is 101.
    seed (int): Random seed for reproducibility. Default is 31.
    freeze_gradients (bool): If True, freezes the gradients of all model parameters. Default is True.
    unfreeze_blocks (int): Currently unused in the function. Reserved for potential future functionality
                           where specific blocks of parameters can be unfrozen.

    Returns:
    tuple: A tuple containing:
        - model (torch.nn.Module): The modified EfficientNet-B2 model with a new classifier head adjusted
                                   for the specified number of classes.
        - transforms (callable): The transformation function required for input images, as recommended by the
                                 pre-trained weights.

    Example:
    ```python
    model, transform = effnetb2(num_classes=100, seed=42, freeze_gradients=True)
    ```

    Notes:
    - The function currently does not use `unfreeze_blocks` but includes it for potential future enhancements.
    - The classifier head of the model is replaced with a new Sequential module that includes dropout and a
      linear layer to adjust to the specified number of classes.
    - The random seed is set for reproducibility of results.
    """
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


def effnetb4(num_classes:int=101, 
             seed:int=31, 
             freeze_gradients:bool=True,
             unfreeze_blocks=0):
    """
    Initializes and configures an EfficientNet-B4 model with options for freezing gradients
    and customizing the classifier head to accommodate a specified number of classes.

    This function sets up an EfficientNet-B4 model pre-trained on the ImageNet-1K dataset, modifies the
    classifier head for the specified number of output classes, and allows for the freezing or partial freezing
    of the model's parameters.

    Parameters:
    num_classes (int): The number of output classes for the new classifier head. Default is 101.
    seed (int): Random seed for reproducibility. Default is 31.
    freeze_gradients (bool): If True, freezes the gradients of the model's parameters. Default is True.
    unfreeze_blocks (int): Currently unused in the function. Reserved for future functionality where specific
                           blocks of parameters can be unfrozen.

    Returns:
    torch.nn.Module: The modified EfficientNet-B4 model with the new classifier head adjusted for the
                     number of specified classes.

    Example:
    ```python
    model = effnetb4(num_classes=100, seed=42, freeze_gradients=True)
    ```

    Notes:
    - The function currently does not use `unfreeze_blocks` but includes it for potential future enhancements.
    - The classifier head of the model is replaced with a new Sequential module that includes dropout and a
      linear layer to adjust to the specified number of classes.
    - The random seed is set for reproducibility of results.
    """
    

    torch.manual_seed(seed)

    #Create model and extract weights/transforms
    weights = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b4(weights=weights)

    #Freeze gradients to avoid modifying the original model    
    if freeze_gradients:
        for param in model.parameters():
            param.requires_grad = False

    #modify classifier model to fit our 
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1792, out_features=num_classes),
    )

    return model, transforms

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


def save_model(model: torch.nn.Module,
               folder_name: str,
               model_name: str):
    """
    Saves the specified PyTorch model's state dict to a file within a specified directory.

    This function ensures the directory exists (creates it if necessary) and then saves the model's
    state dictionary using PyTorch's `torch.save` method.

    Parameters:
    model (torch.nn.Module): The model whose parameters are to be saved.
    folder_name (str): The directory path where the model should be saved. If the directory does not exist,
                       it will be created.
    model_name (str): The filename for the saved model. This should include the file extension,
                      typically '.pt' or '.pth'.

    Example:
    ```python
    model = MyModel()
    save_model(model, 'models', 'my_model.pth')
    ```

    Raises:
    OSError: If the directory specified cannot be created and does not exist already.
    """
    folder_name = Path(folder_name)
    folder_name.mkdir(parents=True, exist_ok=True)
    model_path = folder_name / model_name
    torch.save(obj=model.state_dict(), f=model_path)
