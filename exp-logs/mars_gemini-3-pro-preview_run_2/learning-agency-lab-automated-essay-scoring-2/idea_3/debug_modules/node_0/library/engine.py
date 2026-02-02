import torch
import torch.nn as nn
import numpy as np
import os
import copy
from library.config import Config
from library.utils import compute_qwk


def train_fn(model, data_loader, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        data_loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run training on.
        scheduler (LRScheduler, optional): The learning rate scheduler.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Select loss function based on Config
    criterion = nn.SmoothL1Loss() if Config.use_smooth_l1 else nn.MSELoss()

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)
        # outputs shape is [Batch, 1], targets shape is [Batch]
        outputs = outputs.squeeze(-1)

        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: (average_loss, qwk_score, predictions)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.SmoothL1Loss() if Config.use_smooth_l1 else nn.MSELoss()

    preds = []
    targets_list = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and targets for metric calculation
            preds.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Compute Quadratic Weighted Kappa
    # Input to compute_qwk should be array-like.
    # The function handles rounding of continuous predictions internally if needed.
    val_score = compute_qwk(targets_list, preds)

    return epoch_loss, val_score, np.array(preds)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    scheduler,
    epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device.
        scheduler (LRScheduler): Learning rate scheduler.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model state.

    Returns:
        tuple: (best_model_state_dict, best_score)
    """
    best_score = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_fn(model, train_loader, optimizer, device, scheduler)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss, val_score, _ = eval_fn(model, val_loader, device)
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Val QWK:    {val_score}")  # Printing full precision as requested

        # Early Stopping Logic (Maximize QWK)
        if val_score > best_score:
            print(
                f"Validation Score Improved ({best_score} ---> {val_score}). Saving model..."
            )
            best_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in Validation Score. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val QWK: {best_score}")

    # Load best weights into model before returning
    model.load_state_dict(best_model_wts)

    return best_model_wts, best_score
