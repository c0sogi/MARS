import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple, List, Optional
from library.config import Config
from library.utils import calculate_auc


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: The optimizer.
        device: Computation device.
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate_tta(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, np.ndarray]:
    """
    Evaluates the model using 4-view Test Time Augmentation (TTA).
    Views: Original, Rot90, Rot180, Rot270.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: Computation device.

    Returns:
        Tuple[float, np.ndarray]: Validation AUC score and the array of predictions.
    """
    model.eval()
    y_true = []
    y_pred = []

    # 4-view TTA: 0, 90, 180, 270 degrees
    # Input images are [B, C, H, W]
    # Rotations are on dimensions [2, 3] (H, W)
    rotations = [0, 1, 2, 3]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Store labels
            y_true.extend(labels.numpy())

            # Accumulate probabilities for this batch
            batch_probs_sum = None

            for k in rotations:
                if k == 0:
                    img_aug = images
                else:
                    img_aug = torch.rot90(images, k, [2, 3])

                logits = model(img_aug)
                probs = torch.sigmoid(logits)

                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs

            # Average over 4 views
            avg_probs = batch_probs_sum / len(rotations)
            y_pred.extend(avg_probs.cpu().numpy().ravel())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc = calculate_auc(y_true, y_pred)
    return auc, y_pred


def predict_test_tta(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> np.ndarray:
    """
    Generates predictions for the test set using 4-view TTA.

    Args:
        model: The PyTorch model.
        dataloader: Test dataloader.
        device: Computation device.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    preds = []
    rotations = [0, 1, 2, 3]

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            batch_probs_sum = None

            for k in rotations:
                if k == 0:
                    img_aug = images
                else:
                    img_aug = torch.rot90(images, k, [2, 3])

                logits = model(img_aug)
                probs = torch.sigmoid(logits)

                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs

            avg_probs = batch_probs_sum / len(rotations)
            preds.extend(avg_probs.cpu().numpy().ravel())

    return np.array(preds)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    device: torch.device,
    save_path: str,
):
    """
    Orchestrates the full training loop.

    Implements the strategy:
    - Train for full Config.EPOCHS (no early stopping).
    - Use Cosine Annealing scheduler.
    - Save best model based on TTA Validation AUC.
    """
    best_auc = 0.0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(Config.EPOCHS):
        # 1. Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # 2. Step Scheduler (Cosine Annealing usually steps per epoch)
        if scheduler is not None:
            scheduler.step()

        # 3. Validate with TTA
        val_auc, _ = evaluate_tta(model, val_loader, device)

        # 4. Print Metrics (Full precision as requested)
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val AUC (TTA): {val_auc}"
        )

        # 5. Save Best Model
        # We save if AUC improves, but we continue training until EPOCHS is reached
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            # print(f"New best model saved with AUC: {best_auc}") # Minimal printing

    print(f"Training complete. Best TTA AUC: {best_auc}")
