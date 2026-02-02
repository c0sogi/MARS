import os
import torch
import numpy as np
import torch.nn.functional as F
from library.config import DEVICE, CACHE_DIR
from library.utils import angular_dist_score, vector_to_angles


def train_one_epoch(model, dataloader, optimizer, criterion, device=DEVICE):
    """
    Performs one epoch of training.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size for accuracy)
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    avg_loss = running_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device=DEVICE):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        dict: Dictionary containing 'loss' and 'mae' (Mean Angular Error).
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size

            # Calculate Angular Error
            # 1. Normalize predicted vectors to unit length
            pred_vectors = F.normalize(outputs, p=2, dim=1).cpu().numpy()

            # 2. Convert to spherical coordinates (azimuth, zenith)
            pred_az, pred_zen = vector_to_angles(
                pred_vectors[:, 0], pred_vectors[:, 1], pred_vectors[:, 2]
            )
            pred_angles = np.stack([pred_az, pred_zen], axis=1)

            # 3. Get true angles
            true_angles = targets.cpu().numpy()

            # 4. Compute MAE for this batch
            # angular_dist_score returns the mean error for the batch
            batch_mae = angular_dist_score(true_angles, pred_angles)
            running_mae += batch_mae * batch_size

            num_samples += batch_size

    avg_loss = running_loss / num_samples if num_samples > 0 else 0.0
    avg_mae = running_mae / num_samples if num_samples > 0 else 0.0

    return {"loss": avg_loss, "mae": avg_mae}


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler=None,
    num_epochs=10,
    patience=5,
    device=DEVICE,
    save_path=None,
):
    """
    Main training loop with Early Stopping.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        criterion (Module): Loss function.
        scheduler (lr_scheduler, optional): Learning rate scheduler.
        num_epochs (int): Maximum number of epochs.
        patience (int): Epochs to wait for improvement before stopping.
        device (torch.device): Device to run on.
        save_path (str, optional): Path to save the best model. Defaults to CACHE_DIR/best_model.pth.

    Returns:
        model: The model with the best weights loaded.
        dict: History of training metrics.
    """
    if save_path is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        save_path = os.path.join(CACHE_DIR, "best_model.pth")

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    print(f"Starting training on {device}...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_mae = val_metrics["mae"]

        # Update History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss}, "
            f"Val Loss: {val_loss}, "
            f"Val MAE: {val_mae}"
        )

        # Scheduler Step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # print(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            # print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    if os.path.exists(save_path):
        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model, history
