"""
Ryan Tietjen
Aug 2024
Demo application for a food classificiation demonstration
"""
from model import vit_b_16
from timeit import default_timer as timer
import torch
import os
import gradio as gr 

with open("demo/class_names.txt", 'r') as file:
    class_names = [line.strip() for line in file]


model, transforms = vit_b_16(num_classes=101,
                             seed=31,
                             freeze_gradients=True,
                             unfreeze_blocks=0)

model.load_state_dict(torch.load('demo/vit_b_16_unfreeze_one_encoder_block_10_total_epochs.pth',
                                  weights_only=True))


def predict_single_image(img):
    start_time = timer()

    model.eval()

    #Add batch dim
    img = transforms(img).unsqueeze(dim=0)

    with torch.inference_mode():
        # Obtain prediction logits -> prediction probabilities from image
        logits = model(img)
        probabilities = torch.softmax(logits, dim=1)

        class_probabilities = {}
        for i in range(len(class_names)):
            class_probabilities[class_names[i]] = float(probabilities[0][i])

    end_time = timer()

    pred_time = round(end_time - start_time, 3)

    return class_probabilities, pred_time


title = "Food Image Classification With PyTorch by Ryan Tietjen"
description = f"""
Determines what type of food is presented in a given image. 
This model is capable of classifying [101 different types of food](https://github.com/RyanTietjen/Food-Classifier-pytorch-ver.-/blob/main/demo/class_names.txt) by 
utilizing a [pre-trained Vision Transformer](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights),
and fine-tuning the results for specific food categories.
This model achieved a Top-1 accuracy of 91.55% and a Top-5 accuracy of 98.56%
"""

sample_list = [["demo/samples/" + sample] for sample in os.listdir("demo/samples")]

#Gradio interface
demo = gr.Interface(
    fn=predict_single_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=sample_list,
    title=title,
    description=description,
)

demo.launch(share=True)