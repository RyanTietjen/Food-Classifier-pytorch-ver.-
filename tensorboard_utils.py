"""
Ryan Tietjen
Aug 2024
Contains functions for storing and visualizing results using tensorboard
"""

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

def create_summary_writer(model_name: str, info: str=None):
    """
    Creates and returns a SummaryWriter object for logging and visualizing results using TensorBoard.

    This function sets up a logging directory for storing the results of model runs. It organizes logs 
    into subdirectories within a main 'runs' directory, using the provided model name and additional 
    information string to structure the directory hierarchy. If the specified directories do not exist, 
    they will be created.

    Parameters:
    model_name (str): The name of the model, used as part of the directory path for saving logs.
    info (str, optional): Additional information to further specify the log directory. This could be 
                          details like the date, a specific configuration tag, or experiment identifier.
                          If None, the directory will only use the model_name as the path. Default is None.

    Returns:
    SummaryWriter: A `torch.utils.tensorboard.SummaryWriter` instance initialized with the specified log 
                   directory path. This writer can be used to log data for visualization in TensorBoard.

    Example:
    ```python
    writer = create_summary_writer("my_model", "experiment_1")
    ```

    Notes:
    - The function prints the full path to the directory where the logs will be saved, providing a reference
      for accessing the logs through TensorBoard.
    - The logs are stored under the 'runs' directory in the current working directory, organized by model name
      and optional additional information.
    - This setup facilitates the organization and comparison of different runs and configurations when viewed
      using TensorBoard.
    """
    data = Path("runs")
    data.mkdir(parents=True, exist_ok=True)
    
    log_dir = os.path.join("runs", model_name, info)
    print(f"Saving results to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)
