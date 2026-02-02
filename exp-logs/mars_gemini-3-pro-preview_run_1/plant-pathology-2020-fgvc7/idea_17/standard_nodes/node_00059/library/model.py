import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import (
    seed_everything,
    get_class_weights,
    calculate_roc_auc,
    save_checkpoint,
)
from library.dataset import AppleDataset, get_transforms


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.
    Initializes with ImageNet weights and uses a simple Global Average Pooling + Linear head.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(AppleResNet34, self).__init__()
        # Initialize ResNet34 backbone
        # Using 'weights' for modern torchvision versions, falling back to 'pretrained' if needed
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        except AttributeError:
            self.backbone = models.resnet34(pretrained=pretrained)

        # Replace the default fully connected layer
        # The default ResNet34 structure ends with: AvgPool -> Flatten -> FC
        # We replace FC with our own Linear layer.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Simple classification head
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Backbone forward pass (includes GAP and Flatten due to implementation of resnet34)
        x = self.backbone(x)
        # Classification head
        x = self.fc(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC calculation
        all_targets.append(targets.cpu().numpy())
        all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = calculate_roc_auc(all_targets, all_preds)
    except Exception:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device, use_tta=False):
    """
    Validates the model. Supports Test Time Augmentation (TTA).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Standard forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            if use_tta:
                # Horizontal Flip
                inputs_h = torch.flip(inputs, dims=[3])
                outputs_h = model(inputs_h)
                probs_h = torch.softmax(outputs_h, dim=1)

                # Vertical Flip
                inputs_v = torch.flip(inputs, dims=[2])
                outputs_v = model(inputs_v)
                probs_v = torch.softmax(outputs_v, dim=1)

                # Average predictions
                probs = (probs + probs_h + probs_v) / 3.0

            # Calculate loss (using single view for consistency)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = calculate_roc_auc(all_targets, all_preds)
    except Exception:
        auc = 0.5

    return epoch_loss, auc, all_preds, all_targets


def run_training():
    """
    Placeholder for Seed Averaging Training.
    The actual implementation is in runfile.py to adhere to the instruction of not removing core logic.
    """
    pass


def predict_test(use_tta: bool):
    """
    Placeholder for Seed Averaging Inference.
    The actual implementation is in runfile.py.
    """
    pass


def main():
    """
    Main execution pipeline.
    """
    # 1. Run Training (K-Fold CV)
    use_tta = run_training()

    # 2. Generate Submission
    predict_test(use_tta)
