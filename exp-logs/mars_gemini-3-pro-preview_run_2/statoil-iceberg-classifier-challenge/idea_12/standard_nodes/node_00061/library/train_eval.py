import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import calculate_log_loss, EarlyStopping, seed_everything
from library.model import A2SHN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
        Trains the model for one epoch.

        Args:
            model: The A2SHN model.
    >>>>>>> REPLACE
    <<<<<<< SEARCH
        # Initialize Model
        model = GLPPN()
        model = model.to(device)
    =======
        # Initialize Model
        model = A2SHN()
        model = model.to(device)
    >>>>>>> REPLACE
    <<<<<<< SEARCH
        # Save Model Checkpoint
        model_path = os.path.join(Config.MODEL_DIR, f"glppn_model_fold_{fold_index}.pth")
        torch.save(model.state_dict(), model_path)
    =======
        # Save Model Checkpoint
        model_path = os.path.join(Config.MODEL_DIR, f"a2shn_model_fold_{fold_index}.pth")
        torch.save(model.state_dict(), model_path)
            loader: DataLoader for training data.
            optimizer: The optimizer instance.
            criterion: The loss function (BCEWithLogitsLoss).
            device: The computing device.

        Returns:
            float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The GLPPN model.
        loader: DataLoader for validation data.
        criterion: The loss function.
        device: The computing device.

    Returns:
        tuple: (average_val_loss, log_loss_metric)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            # Forward pass (logits)
            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Convert logits to probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        # Calculate Log Loss metric
        metric_score = calculate_log_loss(y_true, y_pred)
    else:
        metric_score = 0.0

    return avg_loss, metric_score


def run_fold(fold_index, train_loader, val_loader):
    """
    Runs the training and validation loop for a single fold.

    Args:
        fold_index (int): Index of the current fold (for saving models).
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.

    Returns:
        tuple: (best_model, best_log_loss)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting Fold {fold_index} on device: {device}")

    # Initialize Model
    model = A2SHN()
    model = model.to(device)

    # Loss Function (Binary Cross Entropy with Logits)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer ("Low and Slow" Adam)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="min")

    best_log_loss = float("inf")

    for epoch in range(Config.MAX_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_log_loss = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_loss)

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} - "
            f"Train Loss: {train_loss:.16f} - "
            f"Val Loss: {val_loss:.16f} - "
            f"Val Log Loss: {val_log_loss:.16f}"
        )

        # Check Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best weights
    model = early_stopping.restore_best_weights(model)

    # Calculate final metric on best model
    _, best_log_loss = validate(model, val_loader, criterion, device)
    print(f"Fold {fold_index} Finished. Best Val Log Loss: {best_log_loss:.16f}")

    # Save Model Checkpoint
    model_path = os.path.join(Config.MODEL_DIR, f"a2shn_model_fold_{fold_index}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, best_log_loss
