import os
import numpy as np
import torch
import torch.nn as nn
from library.utils import compute_score
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (Loss): Loss function (e.g., BCEWithLogitsLoss).
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        # Model outputs logits. Squeeze to match label shape [Batch]
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def evaluate(model, dataloader, device, use_tta=True):
    """
    Evaluates the model on the validation set using AUC.
    Incorporates Test Time Augmentation (TTA) with 4 views.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (str): Device to evaluate on.
        use_tta (bool): Whether to use Test Time Augmentation.

    Returns:
        float: The AUC score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()

            if use_tta:
                # View 1: Original
                logits_1 = model(images).squeeze(1)
                probs_1 = torch.sigmoid(logits_1)

                # View 2: Horizontal Flip
                images_h = torch.flip(images, dims=[3])
                logits_2 = model(images_h).squeeze(1)
                probs_2 = torch.sigmoid(logits_2)

                # View 3: Vertical Flip
                images_v = torch.flip(images, dims=[2])
                logits_3 = model(images_v).squeeze(1)
                probs_3 = torch.sigmoid(logits_3)

                # View 4: Combined Flip (H + V)
                images_hv = torch.flip(images, dims=[2, 3])
                logits_4 = model(images_hv).squeeze(1)
                probs_4 = torch.sigmoid(logits_4)

                # Average probabilities across all views
                avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0
                batch_preds = avg_probs.cpu().numpy()

            else:
                # Standard inference without TTA
                logits = model(images).squeeze(1)
                probs = torch.sigmoid(logits)
                batch_preds = probs.cpu().numpy()

            all_preds.append(batch_preds)
            all_targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    auc_score = compute_score(y_true, y_pred)
    return auc_score


class EarlyStopping:
    """
    Implements Early Stopping logic to halt training when validation metric stops improving.
    """

    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True  # Improvement (first run)

        if self.mode == "max":
            improvement = score > (self.best_score + self.min_delta)
        else:
            improvement = score < (self.best_score - self.min_delta)

        if improvement:
            self.best_score = score
            self.counter = 0
            return True  # Improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement


def train_fold(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    save_path,
    scheduler=None,
):
    """
    Orchestrates the training loop for a single fold.
    Includes Early Stopping and saving the best model based on TTA Validation AUC.
    """
    early_stopper = EarlyStopping(patience=patience, mode="max")

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # 1. Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 2. Validation Step (with TTA)
        val_auc = evaluate(model, val_loader, device, use_tta=True)

        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Val AUC = {val_auc}")

        # 3. Check Early Stopping & Save
        is_best = early_stopper(val_auc)

        if is_best:
            torch.save(model.state_dict(), save_path)

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load the best weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def predict(model, dataloader, device, use_tta=True):
    """
    Generates predictions for a dataset (e.g., test set).
    Uses the same TTA logic as validation for consistency.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Inference data loader.
        device (str): Device.
        use_tta (bool): Whether to use TTA.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            if use_tta:
                # View 1
                p1 = torch.sigmoid(model(images).squeeze(1))
                # View 2
                p2 = torch.sigmoid(model(torch.flip(images, dims=[3])).squeeze(1))
                # View 3
                p3 = torch.sigmoid(model(torch.flip(images, dims=[2])).squeeze(1))
                # View 4
                p4 = torch.sigmoid(model(torch.flip(images, dims=[2, 3])).squeeze(1))

                avg_probs = (p1 + p2 + p3 + p4) / 4.0
                all_preds.append(avg_probs.cpu().numpy())
            else:
                probs = torch.sigmoid(model(images).squeeze(1))
                all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)
