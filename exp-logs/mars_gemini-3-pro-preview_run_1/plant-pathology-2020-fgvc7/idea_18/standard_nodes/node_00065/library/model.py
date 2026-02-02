import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import pandas as pd
import numpy as np
import os

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    get_class_weights,
    check_initial_loss,
)


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.
    Initializes with ImageNet weights and replaces the head for 4-class classification.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super(AppleResNet34, self).__init__()

        # Load pretrained weights if requested
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet34(weights=weights)

        # Replace the fully connected layer
        # The default ResNet34 fc layer has 512 input features
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # targets are one-hot encoded or probabilities.
        # CrossEntropyLoss expects class indices for hard classification.
        loss = criterion(outputs, torch.argmax(targets, dim=1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions and targets for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Validates the model on the OOB set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, torch.argmax(targets, dim=1))

            running_loss += loss.item() * images.size(0)

            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    val_auc = calculate_roc_auc(all_targets, all_preds)

    return val_loss, val_auc
