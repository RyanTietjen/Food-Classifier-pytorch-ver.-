"""
Ryan Tietjen
Aug 2024
Contains helper functions for training and testing a model
"""

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Executes one training step over a single epoch for a given model using the provided dataloader.

    This function iterates over batches of data from the dataloader, computes the loss using the
    provided loss function, performs backpropagation, and updates the model's weights using the
    specified optimizer. Additionally, it calculates the average loss, top-1 accuracy, and top-5
    accuracy for the epoch.

    Parameters:
        model (torch.nn.Module): The neural network model to be trained.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the training dataset.
        loss_fn (torch.nn.Module): The loss function used to compute the difference between the
            predictions and the actual values.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's weights.
        device (torch.device): The device (CPU or GPU) on which the computations will be performed.

    Returns:
        tuple: A tuple containing the average loss, average top-1 accuracy, and average top-5
            accuracy for the epoch, in that order.

    Notes:
        - The function assumes that the model, loss function, and dataloader are compatible in terms
          of data shapes and types.
        - It is assumed that the model's `.train()` method is called before the training loop to set
          the model to training mode.
    """
    model.train()

    train_loss, acc, top5_acc = 0, 0, 0

    # Loop through batches
    for batch, (X, y) in enumerate(tqdm(dataloader, desc="Training")):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate top 1 accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Calculate top 5 accuracy
        _, top5_pred = y_pred.topk(5, dim=1)
        correct_top5 = top5_pred.eq(y.unsqueeze(1).expand_as(top5_pred))
        top5_acc += (correct_top5.sum().item() / y.size(0))

    avg_loss = train_loss / len(dataloader)
    avg_acc = acc / len(dataloader)
    avg_top5_acc = acc / len(dataloader)

    return avg_loss, avg_acc, avg_top5_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device: torch.device):
    """
    Executes a testing step over a dataset provided by a dataloader to evaluate the model.

    This function iterates over batches of data from the dataloader, computes the loss for each batch
    using the provided loss function, and calculates the top-1 and top-5 accuracies for the model
    without performing backpropagation, as the model is set to evaluation mode. This is intended
    to assess the model's performance on unseen data.

    Parameters:
        model (torch.nn.Module): The neural network model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the testing dataset.
        loss_fn (torch.nn.Module): The loss function to compute errors between predictions and actual
            values.
        device (torch.device): The device (CPU or GPU) on which the computations will be performed.

    Returns:
        tuple: A tuple containing the average loss, average top-1 accuracy, and average top-5
            accuracy for the dataset, in that order.

    Notes:
        - The function ensures no gradients are computed during the evaluation phase to reduce
          memory consumption and improve computational efficiency.
        - The model's `.eval()` method is called to set the model to evaluation
          mode, which turns off specific layers like dropout and batch normalization that behave
          differently during training versus testing.
    """
    model.eval()

    test_loss, acc, top5_acc = 0, 0, 0

    with torch.no_grad():  # No gradient calculations
        for batch, (X, y) in enumerate(tqdm(dataloader, desc="Testing")):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item() 

            # Calculate top 1 accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            acc += (y_pred_class == y).sum().item() / len(y_pred)

            # Calculate top 5 accuracy
            _, top5_pred = y_pred.topk(5, dim=1)
            correct_top5 = top5_pred.eq(y.unsqueeze(1).expand_as(top5_pred))
            top5_acc += (correct_top5.sum().item() / y.size(0))

    avg_loss = test_loss / len(dataloader)
    avg_acc = acc / len(dataloader)
    avg_top5_acc = top5_acc / len(dataloader)

    return avg_loss, avg_acc, avg_top5_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: SummaryWriter,
          verbose: bool):
    """
    Trains and evaluates a PyTorch model over a specified number of epochs, reporting
    performance metrics to the console and optionally to TensorBoard.

    Parameters:
        model (torch.nn.Module): The model to train and evaluate.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for validation or testing data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_fn (torch.nn.Module): Loss function used for training.
        epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the model computation on (e.g., 'cuda' or 'cpu').
        writer (SummaryWriter): TensorBoard writer for logging metrics. If None, logging is skipped.
        verbose (bool): If True, print out detailed logs for each epoch.

    Returns:
        dict: A dictionary containing lists of metrics ('train_loss', 'train_acc', 'train_top5_acc',
              'test_loss', 'test_acc', 'test_top5_acc') recorded during training and testing.

    Notes:
        - This function assumes that the model, dataloaders, loss function, and device are properly
          configured before being passed as arguments.
        - It logs metrics to TensorBoard if a SummaryWriter is provided.
        - Verbose output includes epoch-wise loss and accuracy metrics for both training and testing.
    """
    history = {'train_loss': [], 'train_acc': [], 'train_top5_acc': [], 'test_loss': [], 'test_acc': [], 'test_top5_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc, train_top5_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc, test_top5_acc = test_step(model, test_dataloader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top5_acc'].append(train_top5_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_top5_acc'].append(test_top5_acc)
        
        if writer:
            writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)
            writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)
            writer.add_scalars('Top-5 Accuracy', {'train': train_top5_acc, 'test': test_top5_acc}, epoch)
        
        if verbose:
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train Top-5 Accuracy: {train_top5_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Top-5 Accuracy: {test_top5_acc:.4f}')
        
    return history
