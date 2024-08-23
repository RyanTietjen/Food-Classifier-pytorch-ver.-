"""
Ryan Tietjen
Aug 2024
Temp file for printing model summaries of varies models
"""


import torch
from torchvision.models import efficientnet_b4
from torchinfo import summary
from torchvision.models import efficientnet_b2
from torchvision.models import vit_b_16
model = vit_b_16(pretrained=False)

input_size = (1, 3, 224, 224)

# Print the summary of the model
print(summary(model, input_size=input_size))
