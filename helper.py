import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

device = "cuda" if torch.cuda.is_available() else "cpu"


def return_model():
    model_rn50 = resnet50(weights='IMAGENET1K_V2')
    num_ftrs = model_rn50.fc.in_features

    class_names = ['dogs', 'cats']

    ct = 0
    for child in model_rn50.children():
        ct += 1
        if ct <= 8:
            for param in child.parameters():
                param.requires_grad = False

    model_rn50.fc = nn.Linear(num_ftrs, len(class_names))

    return model_rn50.to(device)


def process_input(image):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mod_input = data_transforms(image)

    return mod_input
