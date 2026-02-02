import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        optimizer: Optimizer.
        device: Device to run on (cuda/cpu).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Ensure targets are (B, 1) for BCEWithLogitsLoss
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (Average Validation Loss, ROC AUC Score).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Calculate AUC
    # Handle potential edge cases where only one class is present
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return val_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for test data.
        device: Device to run on.

    Returns:
        Numpy array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0).flatten()


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    patience,
    save_path,
):
    """
    Runs the full training loop with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model weights.

    Returns:
        The model with the best weights loaded.
    """
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Logic (Maximize AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")

    # Load best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model
