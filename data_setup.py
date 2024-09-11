"""
Ryan Tietjen
Aug-Sep 2024
Helper functions related to setting up the data
"""

import torch
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
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

def get_data_from_S3(dataset, train_ratio=0.8, batch_size=128, num_workers=0, train_transform=None, test_transform=None):
    num_train = int(len(dataset) * train_ratio)
    """
    Splits a dataset into training and testing sets, applies transformations, and creates DataLoaders for both.

    Args:
        dataset (Dataset): The dataset to be split into training and testing sets. This should be a PyTorch Dataset.
        train_ratio (float, optional): The proportion of the dataset to be used for training. Defaults to 0.8.
        batch_size (int, optional): Number of samples per batch to load. Defaults to 128.
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 0.
        train_transform (callable, optional): A function/transform that takes in a sample and returns a transformed version for the training set. Defaults to None.
        test_transform (callable, optional): A function/transform that takes in a sample and returns a transformed version for the testing set. Defaults to None.

    Returns:
        tuple: A tuple containing (train_loader, test_loader) where:
            - train_loader (DataLoader): DataLoader for the training set.
            - test_loader (DataLoader): DataLoader for the testing set.

    Example:
        >>> dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        >>> train_loader, test_loader = get_data_from_S3(
        ...     dataset,
        ...     train_ratio=0.8,
        ...     batch_size=100,
        ...     num_workers=4,
        ...     train_transform=torchvision.transforms.ToTensor(),
        ...     test_transform=torchvision.transforms.ToTensor()
        ... )
        >>> for images, labels in train_loader:
        ...     # Do something with the images and labels

    Note:
        Ensure that the dataset passed as the `dataset` argument supports transformations, which might not be the case for custom datasets or certain pre-packaged torchvision datasets.
    """
    num_test = len(dataset) - num_train

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    # Apply the appropriate transformations if specified
    if train_transform:
        # You need to ensure that the dataset supports transformations
        # For example, if using a custom dataset or torchvision datasets
        train_dataset.dataset.transform = train_transform

    if test_transform:
        test_dataset.dataset.transform = test_transform

    # Create the DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
