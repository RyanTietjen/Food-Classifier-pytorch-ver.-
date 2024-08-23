"""
Ryan Tietjen
Aug 2024
Classifies 101 different types of food images
"""

import torch
import torchvision
from torchvision import transforms
import configparser  # Used in conjunction with the config file

from model_creation import create_model
from model_creation import save_model
from data_setup import get_data
from model_utilization import train
import model_utilization
from tensorboard_utils import create_summary_writer

#parse config file
config = configparser.ConfigParser()
config.read('config.ini')

#Set config settings
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(config["Model Utilization"]["BATCH_SIZE"])
NUM_WORKERS = int(config["Model Utilization"]["NUM_WORKERS"])

#create model
model, transforms = create_model(config)
model = model.to(device)

#load pre exisiting model
if config["Model Utilization"].getboolean("load_exisiting_model"):
    folder_name = "models"
    model_path = '/'.join((folder_name, config["Model Utilization"]["exisiting_model_name"]))
    model.load_state_dict(torch.load(model_path, weights_only=True))


#create custom transforms to give a more diverse set of training data
training_transforms = torchvision.transforms.Compose([
    #https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
    torchvision.transforms.TrivialAugmentWide(), #change color, add noise, flip image, etc.
    transforms, #original model transforms
])

#Setup training and test data
train, test = get_data(training_transforms, transforms, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


#Setup loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), 0.001)
loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

#Create SummaryWriter if we want to track results to TensorBoard
if config["Model Utilization"].getboolean("track_results"):
    writer = create_summary_writer(config["Model Utilization"]["experiment_model_name"],
                                    config["Model Utilization"]["experiment_information"])
else:
    writer = None
    

#Train if applicable
if config["Model Utilization"].getboolean("train"):
    results = model_utilization.train(model=model,
                                    train_dataloader=train,
                                    test_dataloader=test,
                                    optimizer=optimizer,
                                    loss_fn=loss,
                                    epochs=int(config["Model Utilization"]["epochs"]),
                                    device=device,
                                    writer=writer,
                                    verbose = config["Model Utilization"].getboolean("verbose"))
    

#Save if applicable
if config["Model Utilization"].getboolean("save_model"):
    print(f"Saving model {config["Model Utilization"]["model_save_name"]}")
    save_model(model, "models", config["Model Utilization"]["model_save_name"])
