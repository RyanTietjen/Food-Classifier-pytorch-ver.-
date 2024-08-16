import torch
from torchvision.models import efficientnet_b4
from torchinfo import summary
from torchvision.models import efficientnet_b2
from torchvision.models import vit_b_16

# Create a model instance of EfficientNet-B4
model = vit_b_16(pretrained=True)

# Assuming the input size for EfficientNet-B4, which typically is 3 x 380 x 380 for color images
input_size = (1, 3, 224, 224)

# Print the summary of the model
print(summary(model, input_size=input_size))
