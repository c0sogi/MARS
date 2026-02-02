import torch
import torch.nn as nn
import timm
import numpy as np
from library.config import DEVICE, LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE
from library.utils import MetricMonitor, calculate_roc_auc


class ExpertNet(nn.Module):
    """
    ExpertNet wrapper around timm models (EfficientNet-B0).
    Designed for binary classification of MRI slices.
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        num_classes=1,
        in_chans=3,
        drop_rate=DROPOUT_RATE,
    ):
        super(ExpertNet, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )
        self.model_name = model_name

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.model(x)

    def get_optimizer(self, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        """
        Returns the AdamW optimizer configured for this model.
        """
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch in train_loader:
        images = batch["image"].to(device)
        # Ensure labels are (B, 1) to match output logits
        labels = batch["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass (logits)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.get_avg("Loss")


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            metric_monitor.update("Loss", loss.item())

            # Convert logits to probabilities for AUC
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Flatten arrays for metric calculation
            targets.extend(labels.cpu().numpy().flatten())
            preds.extend(probs.flatten())

    auc = calculate_roc_auc(targets, preds)
    return metric_monitor.get_avg("Loss"), auc


def predict(model, test_loader, device):
    """
    Generates predictions for a test loader.
    Returns:
        ids (np.ndarray): Array of BraTS21IDs.
        probs (np.ndarray): Array of predicted probabilities.
    """
    model.eval()
    ids_all = []
    preds_all = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            # BraTS21ID is collated into a Tensor by default
            ids = batch["BraTS21ID"].numpy()

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids_all.extend(ids)
            preds_all.extend(probs)

    return np.array(ids_all), np.array(preds_all)
