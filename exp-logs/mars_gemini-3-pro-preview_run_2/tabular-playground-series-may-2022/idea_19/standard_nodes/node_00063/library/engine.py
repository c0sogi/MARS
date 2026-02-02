import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import save_checkpoint, calculate_auc, print_metric


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (torch.device): The computation device.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move data to device
        continuous = batch["continuous"].to(device)
        sequence = batch["sequence"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Ensure shape (Batch, 1)

        batch_size = continuous.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(continuous, sequence)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (Loss): The loss function.
        device (torch.device): The computation device.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            batch_size = continuous.size(0)

            # Forward pass
            outputs = model(continuous, sequence)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits to get probabilities for AUC
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate results for metric calculation
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc_score = calculate_auc(all_targets, all_preds)

    return avg_loss, auc_score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=Config.NUM_EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Orchestrates the full training process with early stopping and checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Computation device.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.

    Returns:
        float: The best validation AUC achieved.
    """
    # Use BCEWithLogitsLoss as the model outputs logits
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print_metric("Train Loss", train_loss)
        print_metric("Validation Loss", val_loss)
        print_metric("Validation AUC", val_auc)

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model.state_dict(), is_best=True)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc


def predict(model, test_loader, device):
    """
    Generates predictions for the test set and saves them to the submission file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(continuous, sequence)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            ids_list.extend(ids.numpy())
            preds_list.extend(probs.cpu().numpy().flatten())

    # Create DataFrame
    df_submission = pd.DataFrame({"id": ids_list, "target": preds_list})

    # Ensure output directory exists
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    return df_submission
