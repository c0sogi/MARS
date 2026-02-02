import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from library.config import Config
from library.utils import MetricMonitor


def get_model(model_name, pretrained=True, num_classes=1):
    """
    Initializes a model using timm with the specified architecture.
    Modifies the head for binary classification.

    Args:
        model_name (str): Name of the model architecture (e.g., 'convnext_tiny', 'densenet121').
        pretrained (bool): Whether to load ImageNet pretrained weights.
        num_classes (int): Number of output classes (1 for binary classification).

    Returns:
        torch.nn.Module: The initialized PyTorch model.
    """
    # Create model using timm
    # timm handles the num_classes replacement automatically for most models
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        # Mixed precision could be used here, but standard float32 is safer for now
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        # Update metrics
        metric_monitor.update("Loss", loss.item())

        # Calculate batch accuracy for monitoring (threshold 0.5)
        preds = torch.sigmoid(outputs)
        acc = ((preds > 0.5) == (labels > 0.5)).float().mean()
        metric_monitor.update("Acc", acc.item())

    return metric_monitor.avg


def valid_one_epoch(model, loader, criterion, device):
    """
    Validates the model for one epoch. Returns metrics and raw predictions (probabilities).
    """
    model.eval()
    metric_monitor = MetricMonitor()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.sigmoid(outputs)

            metric_monitor.update("Loss", loss.item())

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5  # Handle edge case with single class in batch

    metric_monitor.update("AUC", auc)

    return metric_monitor.avg, all_preds


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (4 views).
    Views: Original, HFlip, VFlip, HVFlip.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            # Loader returns (image, label, id) for test set
            images, _, ids = batch
            images = images.to(device)

            # TTA: 4 views
            # 1. Original
            out1 = torch.sigmoid(model(images))

            # 2. Horizontal Flip
            out2 = torch.sigmoid(model(torch.flip(images, [3])))

            # 3. Vertical Flip
            out3 = torch.sigmoid(model(torch.flip(images, [2])))

            # 4. Horizontal + Vertical Flip
            out4 = torch.sigmoid(model(torch.flip(images, [2, 3])))

            # Average predictions
            avg_preds = (out1 + out2 + out3 + out4) / 4.0

            all_preds.append(avg_preds.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds), all_ids


class StackingMetaLearner:
    """
    Implements the Logistic Regression Meta-Learner for Stacking.
    """

    def __init__(self):
        self.model = LogisticRegression(random_state=Config.SEED)

    def fit(self, X, y):
        """
        Args:
            X (np.array): OOF predictions from base models. Shape (N_samples, N_models).
            y (np.array): Ground truth labels.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Args:
            X (np.array): Test predictions from base models. Shape (N_samples, N_models).
        Returns:
            np.array: Final probabilities.
        """
        # Return probability of positive class
        return self.model.predict_proba(X)[:, 1]


def train_model(model_name, train_loader, val_loader, device, fold_idx):
    """
    Full training routine for a single model fold.
    """
    print(f"--- Training {model_name} | Fold {fold_idx} ---")

    model = get_model(model_name, pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    best_auc = 0.0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"{model_name}_fold_{fold_idx}.pth"
    )

    # Store OOF predictions for the best epoch
    best_oof_preds = None

    # Early stopping counter (optional, but good practice)
    patience = 5
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler=None
        )
        val_metrics, val_preds = valid_one_epoch(model, val_loader, criterion, device)

        # Step scheduler at epoch level
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train: {train_metrics} | "
            f"Val: {val_metrics}"
        )

        # Save best model
        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            best_oof_preds = val_preds
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best AUC for {model_name} Fold {fold_idx}: {best_auc:.5f}")

    # Reload best weights to ensure model state is optimal
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, best_oof_preds
