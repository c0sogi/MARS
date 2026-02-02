import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import save_model


def train_fn(dataloader, model, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch using Mixed Precision (AMP).

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The model to train.
        optimizer: The optimizer.
        device: The device to train on.
        scheduler: Optional learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    final_loss = 0
    scaler = torch.cuda.amp.GradScaler()

    for data in dataloader:
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["labels"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        final_loss += loss.item()

    return final_loss / len(dataloader)


def eval_fn(dataloader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The model to evaluate.
        device: The device to evaluate on.

    Returns:
        tuple: (average_loss, predictions)
    """
    model.eval()
    final_loss = 0
    final_preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.long)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, targets)

            final_loss += loss.item()

            # Apply softmax to get probabilities
            preds = torch.softmax(outputs, dim=1)
            final_preds.append(preds.cpu().numpy())

    avg_loss = final_loss / len(dataloader)
    final_preds = np.vstack(final_preds)

    return avg_loss, final_preds


def predict_fn(dataloader, model, device):
    """
    Generates predictions for the test set (no labels).

    Args:
        dataloader: PyTorch DataLoader for test data.
        model: The model to use for prediction.
        device: The device to predict on.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()
    final_preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)

            preds = torch.softmax(outputs, dim=1)
            final_preds.append(preds.cpu().numpy())

    final_preds = np.vstack(final_preds)
    return final_preds


def run_training(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    num_epochs,
    patience,
    fold,
    model_name,
    scheduler=None,
):
    """
    Orchestrates the training process with Early Stopping and Model Checkpointing.

    Args:
        model: The model to train.
        train_dataloader: DataLoader for training.
        val_dataloader: DataLoader for validation.
        optimizer: Optimizer.
        device: Device.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        fold: Current fold number (for logging and saving).
        model_name: Name prefix for saving the model.
        scheduler: Optional scheduler.

    Returns:
        tuple: (best_val_loss, best_val_predictions)
    """
    best_loss = np.inf
    best_preds = None
    patience_counter = 0

    # Ensure model is on the correct device
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_fn(train_dataloader, model, optimizer, device, scheduler)
        val_loss, val_preds = eval_fn(val_dataloader, model, device)

        # Print metrics with full precision
        print(
            f"Fold {fold} | Epoch {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_preds = val_preds
            patience_counter = 0
            # Save best model state
            save_name = f"{model_name}_fold_{fold}.bin"
            save_model(model, save_name, model_type="torch")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_loss, best_preds
