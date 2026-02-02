import os
import torch
import numpy as np
import pandas as pd
from library.utils import compute_roc_auc


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The IAPEModel instance.
        dataloader: DataLoader for training data.
        optimizer: The optimizer (AdamW).
        scheduler: The learning rate scheduler (OneCycleLR).
        criterion: The loss function (BCEWithLogitsLoss).
        device: The device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Unpack batch from ManufacturingDataset
        x_cont = batch["continuous"].to(device)
        x_cat = batch["categorical"].to(device)
        y = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass: returns [batch_size, num_streams]
        logits = model(x_cont, x_cat)

        # Calculate loss: Sum of BCE losses for each stream
        loss = 0
        num_streams = logits.shape[1]
        for i in range(num_streams):
            loss += criterion(logits[:, i], y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The IAPEModel instance.
        dataloader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to run evaluation on.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x_cont = batch["continuous"].to(device)
            x_cat = batch["categorical"].to(device)
            y = batch["target"].to(device)

            logits = model(x_cont, x_cat)

            # Compute loss for monitoring (Sum of streams)
            loss = 0
            num_streams = logits.shape[1]
            for i in range(num_streams):
                loss += criterion(logits[:, i], y)

            running_loss += loss.item()
            num_batches += 1

            # Predictions: Arithmetic mean of probabilities from all 5 streams
            probs = torch.sigmoid(logits).mean(dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        auc = compute_roc_auc(all_targets, all_preds)
    else:
        auc = 0.0

    return avg_loss, auc


def predict(model, dataloader, device):
    """
    Generates probability predictions for the test set.

    Args:
        model: The trained IAPEModel.
        dataloader: DataLoader for test data.
        device: The device to run inference on.

    Returns:
        np.array: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_cont = batch["continuous"].to(device)
            x_cat = batch["categorical"].to(device)

            logits = model(x_cont, x_cat)

            # Predictions: Arithmetic mean of probabilities from all 5 streams
            probs = torch.sigmoid(logits).mean(dim=1)
            all_preds.append(probs.cpu().numpy())

    if len(all_preds) > 0:
        return np.concatenate(all_preds)
    else:
        return np.array([])


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training loop, including validation, early stopping, and model checkpointing.

    Args:
        model: The model to train.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer instance.
        scheduler: Scheduler instance.
        criterion: Loss function.
        device: Device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.

    Returns:
        tuple: (trained_model, best_val_auc)
    """
    best_auc = -float("inf")
    patience_counter = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics in full precision
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load the best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model, best_auc


def generate_submission(model, test_loader, test_ids, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: Trained model.
        test_loader: DataLoader for test data.
        test_ids: Array-like of test IDs corresponding to the loader data.
        device: Device.
        output_path: Path to save the submission CSV.
    """
    preds = predict(model, test_loader, device)

    if len(preds) != len(test_ids):
        raise ValueError(
            f"Length mismatch: generated {len(preds)} predictions for {len(test_ids)} IDs."
        )

    submission = pd.DataFrame({"id": test_ids, "target": preds})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
