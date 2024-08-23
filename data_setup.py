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
    """
    Prepares and returns data loaders for the training and testing datasets using the Food-101 dataset.

    This function handles the downloading and loading of the Food-101 dataset into PyTorch DataLoader objects
    with specified transformations for training and testing. The datasets are stored in a directory specified
    by the `path` argument. If the directory does not exist, it will be created. The data is then loaded using
    the specified batch size, and the number of worker threads for loading data can be configured.

    Parameters:
    training_transforms (callable): The transformations to apply to the training data.
    testing_transforms (callable): The transformations to apply to the testing data.
    path (str, optional): The base directory where the dataset will be stored. Default is "data".
    batch_size (int, optional): The number of samples in each batch of data. Default is 32.
    num_workers (int, optional): The number of subprocesses to use for data loading. Default is 0, which means
                                 that the data will be loaded in the main process.

    Returns:
    tuple: A tuple containing:
        - train (torch.utils.data.DataLoader): DataLoader for the training dataset.
        - test (torch.utils.data.DataLoader): DataLoader for the testing dataset.

    Example:
    ```python
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    train_loader, test_loader = get_data(train_transform, test_transform)
    ```

    Notes:
    - The Food-101 dataset is automatically downloaded if not already present in the specified directory.
    - The `shuffle=True` parameter for the training DataLoader ensures that the dataset is shuffled for
      each epoch, providing randomization that can help improve model generalization.
    - The `shuffle=False` parameter for the testing DataLoader ensures that the order of the test dataset
      is consistent, allowing for consistent evaluation metrics.
    """
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
