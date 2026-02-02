import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import quadratic_weighted_kappa, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        device: Computation device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # BCE Loss for each of the 4 ordinal units
    criterion = nn.BCELoss(reduction="none")

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)

        # Calculate loss
        # outputs: (B, 4), targets: (B, 4)
        # We sum the BCE loss across the 4 units for each sample, then average over the batch
        loss_per_sample = criterion(outputs, targets).sum(dim=1)
        loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        device: Computation device.

    Returns:
        tuple: (average_val_loss, qwk_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCELoss(reduction="none")

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)

            # Loss calculation
            loss_per_sample = criterion(outputs, targets).sum(dim=1)
            loss = loss_per_sample.mean()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Decoding Ordinal Predictions
            # Sum the probabilities of the 4 units to get a score [0, 4]
            pred_scores = outputs.sum(dim=1)
            # Round to nearest integer to get class label
            pred_labels = pred_scores.round().cpu().numpy().astype(int)

            # Decode Targets
            # Summing the binary target vector recovers the integer class label
            target_labels = targets.sum(dim=1).cpu().numpy().astype(int)

            all_preds.extend(pred_labels)
            all_targets.extend(target_labels)

    val_loss = running_loss / dataset_size

    # Calculate Quadratic Weighted Kappa
    qwk = quadratic_weighted_kappa(all_targets, all_preds)

    return val_loss, qwk


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience
):
    """
    Main training loop with early stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Computation device.
        epochs: Total number of epochs.
        patience: Early stopping patience.

    Returns:
        float: Best validation QWK score.
    """
    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_qwk = validate(model, val_loader, device)

        # Step the scheduler (CosineAnnealingLR steps per epoch)
        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val QWK: {val_qwk}"
        )

        # Early Stopping and Checkpointing based on QWK
        if val_qwk > best_score + Config.early_stopping_min_delta:
            best_score = val_qwk
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                filename="best_model.pth",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_score


def predict(model, test_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model: PyTorch model.
        test_loader: Test DataLoader.
        device: Computation device.

    Returns:
        pd.DataFrame: DataFrame containing 'id_code' and 'diagnosis'.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, id_codes in test_loader:
            images = images.to(device)

            outputs = model(images)

            # Decode predictions
            pred_scores = outputs.sum(dim=1)
            pred_labels = pred_scores.round().cpu().numpy().astype(int)

            for id_code, pred in zip(id_codes, pred_labels):
                results.append({"id_code": id_code, "diagnosis": pred})

    df_submission = pd.DataFrame(results)
    return df_submission


def generate_submission(model, test_loader, device):
    """
    Runs inference and saves the submission file.

    Args:
        model: PyTorch model.
        test_loader: Test DataLoader.
        device: Computation device.
    """
    print("Generating submission...")
    df_sub = predict(model, test_loader, device)

    output_path = Config.submission_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
