import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import MetricMonitor


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): PyTorch optimizer.
        device (str): Device to train on ('cpu' or 'cuda').
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch in train_loader:
        # Move data to device
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update metrics (weight by batch size for accurate epoch average)
        batch_size = images.size(0)
        metric_monitor.update("Loss", loss.item(), n=batch_size)

    return metric_monitor.get_avg("Loss")


def evaluate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    metric_monitor = MetricMonitor()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            # Update metrics
            batch_size = images.size(0)
            metric_monitor.update("Loss", loss.item(), n=batch_size)

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(probs.cpu().numpy())

    # Concatenate results from all batches
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    # Calculate ROC AUC
    # Handle edge case where validation batch might only have one class
    try:
        auc_score = roc_auc_score(all_targets, all_predictions)
    except ValueError:
        auc_score = 0.5

    return metric_monitor.get_avg("Loss"), auc_score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Main training loop with Early Stopping.

    Args:
        model (nn.Module): The PyTorch model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): PyTorch optimizer.
        device (str): Device to train on.
        epochs (int): Maximum number of epochs.
        save_path (str): Path to save the best model weights.

    Returns:
        float: Best validation AUC score achieved.
    """
    # Define Loss function with positive weight for class imbalance
    # We create the tensor on the correct device
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Scheduler to refine convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1
    )

    best_auc = 0.0
    patience = 3
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        # Train step
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validation step
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch}: Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best AUC! Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement in AUC. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_auc


def predict_and_submit(
    model, test_loader, device, submission_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for test data.
        device (str): Device to run inference on.
        submission_path (str): Path to save the submission CSV.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    model.eval()

    image_names = []
    probabilities = []

    print("Starting inference on test set...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            names = batch["image_name"]

            logits = model(images)
            probs = torch.sigmoid(logits)

            image_names.extend(names)
            probabilities.extend(probs.cpu().numpy().flatten())

    # Create DataFrame conforming to submission format
    df_sub = pd.DataFrame({"image_name": image_names, "target": probabilities})

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Save to CSV
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return df_sub
