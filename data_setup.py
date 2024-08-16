"""
Ryan Tietjen
Aug 2024
Helper functions related to setting up the data
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_data(training_transforms,
             testing_transforms,
             path:str = "data",
             batch_size:int = 32,
             num_workers:int = 0):
    #get data
    data = Path(path)
    data.mkdir(parents=True, exist_ok=True)

    train_data = datasets.Food101(root=data,
                                split="train", 
                                transform=training_transforms, 
                                download=True) 
    test_data = datasets.Food101(root=data,
                                split="test",
                                transform=testing_transforms, # no need for TrivialAugmentWide transforms on testing data
                                download=True)
    
    #Turn to Dataloaders
    train = torch.utils.data.DataLoader(train_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)
    test = torch.utils.data.DataLoader(test_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

    return train, test
