import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utils import AverageMeter, quadratic_weighted_kappa


def train_one_epoch(
    model, dataloader, optimizer, device, accumulation_steps=1, epoch=0
):
    """
    Trains the model for one epoch using Gradient Accumulation.

    Args:
        model: The neural network model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Torch device (cpu or cuda).
        accumulation_steps: Number of steps to accumulate gradients before updating.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    num_batches = len(dataloader)

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1) for regression

        # Forward pass
        outputs = model(images)
        loss = nn.MSELoss()(outputs, targets)

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Step optimizer
        # Perform step if accumulation cycle is complete or if it's the last batch
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()

        # Update statistics (multiply back to get actual loss for logging)
        loss_meter.update(loss.item() * accumulation_steps, images.size(0))

    return loss_meter.avg


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: Validation dataloader.
        device: Torch device.

    Returns:
        tuple: (Average Loss, Quadratic Weighted Kappa Score)
    """
    model.eval()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = nn.MSELoss()(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))

            # Store predictions and targets for metric calculation
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate QWK
    # The utility function handles rounding and clipping internally
    qwk = quadratic_weighted_kappa(all_targets, all_preds)

    return loss_meter.avg, qwk


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=10,
    accumulation_steps=1,
    patience=5,
    save_path="best_model.pth",
):
    """
    Runs the full training loop with Early Stopping.

    Args:
        model: The neural network model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer instance.
        device: Torch device.
        epochs: Maximum number of epochs.
        accumulation_steps: Gradient accumulation steps.
        patience: Epochs to wait for improvement before stopping.
        save_path: Path to save the best model.

    Returns:
        float: Best validation QWK score.
    """
    best_qwk = -float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, accumulation_steps, epoch
        )
        val_loss, val_qwk = evaluate(model, val_loader, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val QWK = {val_qwk}"
        )

        # Early Stopping Logic based on QWK
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved with QWK: {best_qwk}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val QWK: {best_qwk}")
    return best_qwk


def predict_and_submit(
    model,
    test_loader,
    device,
    submission_path="./submission/submission.csv",
    sample_sub_path="./input/sample_submission.csv",
):
    """
    Generates predictions for the test set and saves to submission.csv.

    Args:
        model: The trained neural network model.
        test_loader: Test dataloader.
        device: Torch device.
        submission_path: Path to save the submission CSV.
        sample_sub_path: Path to the sample submission file to ensure format/order.
    """
    model.eval()
    all_preds = []

    print("Generating predictions for test set...")

    # Inference
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    # Post-processing: Round to nearest integer and clip to [0, 4]
    predictions = np.rint(all_preds).clip(0, 4).astype(int).flatten()

    # Prepare Submission DataFrame
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # We attempt to load the sample submission to preserve ID order.
    # If not available, we rely on the metadata used to create the test loader.
    if os.path.exists(sample_sub_path):
        sub_df = pd.read_csv(sample_sub_path)
    else:
        # Fallback: Load from metadata
        meta_test_path = "./metadata/test.csv"
        if os.path.exists(meta_test_path):
            sub_df = pd.read_csv(meta_test_path)
            # Ensure only relevant columns are kept if metadata has extra info
            if "id_code" in sub_df.columns:
                sub_df = sub_df[["id_code"]].copy()
        else:
            raise FileNotFoundError(
                "Neither sample_submission.csv nor metadata/test.csv found."
            )

    # Verify lengths
    if len(predictions) != len(sub_df):
        print(
            f"Warning: Prediction count ({len(predictions)}) does not match submission row count ({len(sub_df)})."
        )
        # Truncate or pad to avoid crash, though this indicates a pipeline issue
        if len(predictions) > len(sub_df):
            predictions = predictions[: len(sub_df)]
        else:
            # Pad with zeros
            padding = np.zeros(len(sub_df) - len(predictions), dtype=int)
            predictions = np.concatenate([predictions, padding])

    sub_df["diagnosis"] = predictions

    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
