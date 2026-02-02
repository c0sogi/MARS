import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import DEVICE
from library.utils import save_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Computation device.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in dataloader:
        images = images.to(device)
        # Ensure targets are (N, 1) for BCEWithLogitsLoss
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions (probabilities) and targets for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Handle edge cases where a batch/epoch might contain only one class
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    save_name,
):
    """
    Runs the full training loop with early stopping and checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs: Maximum number of epochs.
        device: Computation device.
        save_name: Filename for saving the best checkpoint.

    Returns:
        float: Best validation AUC achieved.
    """
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss}, Train AUC: {train_auc}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # Save best model based on Validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                save_name,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_auc


def predict_tta(model, dataloader, device):
    """
    Performs inference with Test Time Augmentation (Original, Horizontal Flip, Vertical Flip).
    Assumes the model is in inference mode (preferably re-parameterized/fused).

    Args:
        model: The PyTorch model.
        dataloader: Test dataloader.
        device: Computation device.

    Returns:
        tuple: (ids, predictions) where predictions are averaged probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            # CactusDataset returns (image, target), we only need image
            images, _ = batch
            images = images.to(device)

            # 1. Original Prediction
            outputs_orig = model(images)
            probs_orig = torch.sigmoid(outputs_orig)

            # 2. Horizontal Flip Prediction (flip width dim=3)
            images_h = torch.flip(images, dims=[3])
            outputs_h = model(images_h)
            probs_h = torch.sigmoid(outputs_h)

            # 3. Vertical Flip Prediction (flip height dim=2)
            images_v = torch.flip(images, dims=[2])
            outputs_v = model(images_v)
            probs_v = torch.sigmoid(outputs_v)

            # Average the probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            all_preds.append(avg_probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Retrieve IDs from the dataset (assuming sequential access with shuffle=False)
    all_ids = dataloader.dataset.ids

    return all_ids, all_preds
