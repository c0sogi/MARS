import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from library.config import Config
from library.utils import AverageMeter, EarlyStopping


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function.
        device (torch.device): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for spectrogram, features, target in dataloader:
        # Move inputs and target to device
        spectrogram = spectrogram.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)

        # Target shape from dataset is (Batch), model output is (Batch, 1)
        target = target.to(device, non_blocking=True).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(spectrogram, features)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update tracking
        loss_meter.update(loss.item(), spectrogram.size(0))

    return loss_meter.avg


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for spectrogram, features, target in dataloader:
            spectrogram = spectrogram.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).unsqueeze(1)

            output = model(spectrogram, features)
            loss = criterion(output, target)

            loss_meter.update(loss.item(), spectrogram.size(0))

    return loss_meter.avg


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    config,
):
    """
    Orchestrates the full training process with logging, scheduler, and early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        optimizer (Optimizer): Optimizer instance.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Computation device.
        epochs (int): Number of epochs to train.
        config (Config): Configuration class with paths and params.

    Returns:
        nn.Module: The model with the best weights loaded.
    """
    criterion = nn.L1Loss()

    # Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=config.MODEL_SAVE_PATH,
    )

    for epoch in range(epochs):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation Step
        val_loss = evaluate(model, val_loader, criterion, device)

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Scheduler Step (ReduceLROnPlateau monitors validation loss)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early Stopping Check
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model weights found during training
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading best model from {config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    return model


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set, applies inverse scaling,
    and saves the results to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Computation device.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    predictions = []

    # Load target scalers for inverse transformation
    # These are saved by the Dataset class during training initialization
    if os.path.exists(Config.TARGET_MEAN_PATH) and os.path.exists(
        Config.TARGET_STD_PATH
    ):
        target_mean = np.load(Config.TARGET_MEAN_PATH)
        target_std = np.load(Config.TARGET_STD_PATH)
    else:
        # Fallback (e.g., if running inference without prior training in this session)
        print("Warning: Target scaler files not found. Using identity scaling.")
        target_mean = 0.0
        target_std = 1.0

    print("Generating predictions...")
    with torch.no_grad():
        for spectrogram, features, _ in test_loader:
            spectrogram = spectrogram.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)

            # Forward pass (Output is scaled)
            output = model(spectrogram, features)

            # Move to CPU and flatten to 1D array
            output_np = output.cpu().numpy().flatten()

            # Inverse Scale: original = (scaled * std) + mean
            output_unscaled = (output_np * target_std) + target_mean

            predictions.extend(output_unscaled)

    # Retrieve Segment IDs from the dataset
    # The DataLoader preserves order if shuffle=False (standard for test loaders)
    # We access the underlying dataframe of the dataset
    segment_ids = test_loader.dataset.df["segment_id"].values

    # Validation check
    if len(segment_ids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(segment_ids)} IDs vs {len(predictions)} predictions."
        )

    # Create Submission DataFrame
    df_submission = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
