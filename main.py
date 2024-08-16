"""
Ryan Tietjen
Aug 2024
Classifies 101 different types of food images
"""

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import Dataset
import zipfile
from pathlib import Path
from PIL import Image
import configparser  # Used in conjunction with the config file
import os

from model_creation import create_model
from model_creation import save_model
from data_setup import get_data
from model_utilization import train
import model_utilization

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
BATCH_SIZE = 128
NUM_WORKERS = 0

#parse config file
config = configparser.ConfigParser()
config.read('config.ini')
#create model
model, transforms = create_model(config)
model = model.to(device)
if config["Model Utilization"].getboolean("load_exisiting_model"):
    folder_name = "models"
    model_path = '/'.join((folder_name, config["Model Utilization"]["exisiting_model_name"]))
    model.load_state_dict(torch.load(model_path, weights_only=True))

# model = torch.compile(model)

#create custom transforms to give a more diverse set of training data
training_transforms = torchvision.transforms.Compose([
    #https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
    torchvision.transforms.TrivialAugmentWide(), #change color, add noise, flip image, etc.
    transforms, #original model transforms
])

NUM_WORKERS = 2 if os.cpu_count() <= 4 else 4
train, test = get_data(training_transforms, transforms, batch_size=BATCH_SIZE, num_workers=0)

optimizer = torch.optim.Adam(model.parameters(), 0.001)
loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

if config["Model Utilization"].getboolean("train"):
    results = model_utilization.train(model=model,
                                    train_dataloader=train,
                                    test_dataloader=test,
                                    optimizer=optimizer,
                                    loss_fn=loss,
                                    epochs=1,
                                    device=device)

save_model(model, "models", config["Model Utilization"]["model_save_name"])
