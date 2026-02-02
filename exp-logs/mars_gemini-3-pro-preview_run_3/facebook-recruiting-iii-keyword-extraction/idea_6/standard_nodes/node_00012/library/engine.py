import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library import config, utils

# Initialize logger
logger = utils.get_logger("engine")


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """
    Trains the model for one epoch using Automatic Mixed Precision.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to train on ('cuda' or 'cpu').
        scaler (GradScaler): Scaler for AMP.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Forward Pass
        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scaled Backward Pass
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (str): Device to evaluate on.
        threshold (float): Probability threshold for binary classification.

    Returns:
        tuple: (Average Loss, F1 Score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = len(dataloader)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=True):
                logits = model(inputs)
                loss = criterion(logits, targets)

            running_loss += loss.item()

            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / num_batches

    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # Calculate F1 Score
    f1 = utils.calculate_f1_score(y_true, y_pred, average="micro")

    return avg_loss, f1


def predict(model, dataloader, device):
    """
    Generates raw logits for the provided dataset.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Data loader (usually test set).
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of raw logits.
    """
    model.eval()
    all_logits = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=True):
                logits = model(inputs)

            all_logits.append(logits.cpu().numpy())

    return np.vstack(all_logits)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Orchestrates the full training process with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (str): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.

    Returns:
        float: Best Validation F1 Score achieved.
    """
    scaler = torch.cuda.amp.GradScaler()
    best_f1 = -1.0
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        logger.info(f"Epoch {epoch + 1} completed in {elapsed}s")
        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Val Loss: {val_loss}")
        logger.info(f"Val F1: {val_f1}")

        # Early Stopping Logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    return best_f1


def find_optimal_threshold(model, val_loader, device):
    """
    Scans different probability thresholds to find the one that maximizes F1 score on the validation set.

    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation data.
        device (str): Device.

    Returns:
        float: Optimal threshold.
    """
    model.eval()
    all_probs = []
    all_targets = []

    logger.info("Finding optimal threshold...")

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(inputs)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets).astype(np.int8)

    best_thresh = 0.5
    best_f1 = 0.0

    # Search range from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.95, 0.05)
    for thresh in thresholds:
        y_pred = (y_prob > thresh).astype(np.int8)
        f1 = utils.calculate_f1_score(y_true, y_pred, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    logger.info(f"Optimal threshold: {best_thresh} (F1: {best_f1})")
    return best_thresh


def generate_submission(
    model, test_loader, tag_encoder, device, output_path, threshold=None
):
    """
    Generates the submission CSV file by predicting on the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        tag_encoder (TagEncoder): Encoder to convert indices back to tag strings.
        device (str): Device.
        output_path (str): Path to save the submission CSV.
        threshold (float, optional): Threshold for prediction. Defaults to 0.5 if None.
    """
    logger.info("Generating submission...")

    # Retrieve all IDs from the loader (order is preserved)
    all_ids = []
    for _, ids in test_loader:
        all_ids.extend(ids.numpy())

    # Get raw logits
    logits = predict(model, test_loader, device)
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid

    if threshold is None:
        threshold = 0.5

    # Convert probabilities to tag strings
    pred_tags_list = []
    classes = tag_encoder.classes_

    for i in range(len(probs)):
        row_probs = probs[i]
        indices = np.where(row_probs > threshold)[0]

        # Ensure at least one tag is predicted if all probs are low
        if len(indices) == 0:
            indices = [np.argmax(row_probs)]

        tags = classes[indices]
        pred_tags_list.append(" ".join(tags))

    # Create DataFrame and save
    df = pd.DataFrame({"Id": all_ids, "Tags": pred_tags_list})

    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
