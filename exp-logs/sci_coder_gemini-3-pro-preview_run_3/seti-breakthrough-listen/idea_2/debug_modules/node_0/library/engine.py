import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from library.config import Config
from library.utils import AverageMeter, get_score
from library.dataset import mixup_data, mixup_criterion


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The computing device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if Config.USE_MIXUP:
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, Config.MIXUP_ALPHA, device
            )
            outputs = model(images)
            loss = mixup_criterion(
                criterion, outputs, targets_a.view(-1, 1), targets_b.view(-1, 1), lam
            )
        else:
            outputs = model(images)
            loss = criterion(outputs, targets.view(-1, 1))

        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def valid_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The computing device.

    Returns:
        tuple: (Average validation loss, Validation AUC score)
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    valid_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets.view(-1, 1))

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            preds.append(probs.detach().cpu().numpy())
            valid_targets.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds)
    valid_targets = np.concatenate(valid_targets)

    auc = get_score(valid_targets, preds)

    return losses.avg, auc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs=Config.NUM_EPOCHS,
    patience=3,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        device: Computing device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
    """
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Save best model logic
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model found! Saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping logic
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {epoch} epochs. Best AUC: {best_auc}"
            )
            break


def inference(model, loader, device):
    """
    Generates predictions for a dataset.

    Args:
        model: The PyTorch model.
        loader: The DataLoader (test set).
        device: Computing device.

    Returns:
        np.array: Predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds.append(probs.detach().cpu().numpy())

    return np.concatenate(preds)


def predict_and_submit(
    model, test_loader, device, submission_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The trained PyTorch model.
        test_loader: Test DataLoader.
        device: Computing device.
        submission_path: Path to save the CSV.
    """
    print("Generating predictions for test set...")
    predictions = inference(model, test_loader, device)

    # Load the sample submission to get IDs
    # We assume the test_loader yields data in the same order as the test.csv metadata
    # which is derived from sample_submission.csv
    df_sub = pd.read_csv(Config.TEST_CSV)

    # Ensure lengths match
    if len(df_sub) != len(predictions):
        raise ValueError(
            f"Length mismatch: Test set has {len(df_sub)} rows but generated {len(predictions)} predictions."
        )

    # Flatten predictions if necessary (e.g. from (N, 1) to (N,))
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    df_sub["target"] = predictions

    # The submission format requires 'id' and 'target'
    # The test.csv from metadata already has these columns (plus file_path)
    output_df = df_sub[["id", "target"]]

    output_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
