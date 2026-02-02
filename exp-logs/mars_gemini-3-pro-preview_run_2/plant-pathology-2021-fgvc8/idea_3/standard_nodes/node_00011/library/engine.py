import os
import copy
import time
import torch
import torch.nn as nn
import numpy as np
from library.config import CFG
from library.utils import get_score


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    """
    Trains the model for one epoch using gradient accumulation.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        dataloader: Training DataLoader.
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    accum_iter = CFG.gradient_accumulation_steps

    # Zero gradients at the start
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        loss = loss / accum_iter
        loss.backward()

        # Update weights every accum_iter steps or at the end of the epoch
        if ((step + 1) % accum_iter == 0) or ((step + 1) == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        # Track loss (scale back up for reporting)
        running_loss += (loss.item() * accum_iter) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        device: Device to validate on.

    Returns:
        tuple: (Average validation loss, Validation F1 Score)
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits for metric calculation
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_loss = running_loss / dataset_size
    val_f1 = get_score(all_targets, all_preds)

    return epoch_loss, val_f1


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    model_name="model",
    patience=7,
):
    """
    Runs the full training loop with early stopping and checkpointing.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Total number of epochs.
        model_name: Name prefix for saving the model.
        patience: Number of epochs to wait for improvement before early stopping.

    Returns:
        tuple: (Best F1 Score, Path to saved best model)
    """
    # Initialize best score and weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -np.inf

    # Early stopping counter
    trigger_times = 0

    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)
    save_path = os.path.join(CFG.output_dir, f"{model_name}_best.pth")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss, val_f1 = valid_one_epoch(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Print Metrics (Full Precision)
        print(
            f"Epoch: {epoch+1}/{epochs} | Time: {int(epoch_mins)}m {int(epoch_secs)}s"
        )
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F1: {val_f1}")

        # Checkpointing & Early Stopping
        if val_f1 > best_f1:
            print(f"Validation F1 Improved ({best_f1} ---> {val_f1})")
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"Model Saved to {save_path}")
            trigger_times = 0  # Reset early stopping
        else:
            trigger_times += 1
            print(f"EarlyStopping counter: {trigger_times} out of {patience}")

            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        print("-" * 30)

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)

    return best_f1, save_path
