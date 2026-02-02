import os
import torch
import torch.nn as nn
from library.config import Config
from library.utils import set_seed


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader providing training batches.
        optimizer: The optimizer for weight updates.
        scheduler: Learning rate scheduler (optional).
        device: The device to run computation on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move batch inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Handle token_type_ids if present (common in BERT/MuRIL)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Move labels to device
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # Forward pass
        # The model computes loss automatically if start/end positions are provided
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_one_epoch(model, dataloader, device):
    """
    Performs one epoch of validation.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader providing validation batches.
        device: The device to run computation on.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # We can only compute loss if labels are present.
            # In K-Fold CV, the validation set is a subset of train, so labels exist.
            if "start_positions" in batch and "end_positions" in batch:
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
            else:
                # If using the provided 'val.csv' via get_val_data(), labels might be missing.
                # In that specific inference case, loss cannot be computed.
                continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
    patience=Config.PATIENCE,
):
    """
    Orchestrates the training loop with Early Stopping.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Maximum number of epochs.
        save_path: File path to save the best model checkpoint.
        patience: Number of epochs to wait for improvement before stopping.

    Returns:
        model: The model loaded with the best weights found during training.
    """
    best_val_loss = float("inf")
    patience_counter = 0

    # Ensure the directory for the checkpoint exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpoint logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Early Stopping logic
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load the best model state before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model
