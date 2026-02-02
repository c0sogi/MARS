import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
    """
    Constructs and returns a model customized for the specific number of classes.
    Supports ResNet-18 and EfficientNet-B0 based on Config.MODEL_NAME.
    """
    if Config.MODEL_NAME == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        model = models.efficientnet_b0(weights=weights)

        # EfficientNet classifier is a Sequential block. The linear layer is at index 1.
        # (classifier): Sequential(
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        # Default to ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
