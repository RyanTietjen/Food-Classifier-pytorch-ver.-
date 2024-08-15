"""
Ryan Tietjen
Aug 2024
Detects 101 different types of food images
"""

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import Dataset
from torchinfo import summary
import requests
import zipfile
from pathlib import Path
from PIL import Image
import configparser  # Used in conjunction with the config file

from model_creation import create_model

#Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#parse config file
config = configparser.ConfigParser()
config.read('config.ini')

#create model
model, transforms = create_model(config)

#create custom transforms to give a more diverse set of training data
training_transforms = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(), #change color, add noise, flip image, etc.
    transforms, #original model transforms
])


#get data
data = Path("data")
train_data = datasets.Food101(root=data,
                              split="train", 
                              transform=training_transforms, 
                              download=True) 
test_data = datasets.Food101(root=data,
                             split="test",
                             transform=transforms, # no need for TrivialAugmentWide transforms on testing data
                             download=True)