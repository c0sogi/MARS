import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one epoch of training.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Computation device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        # BCEWithLogitsLoss expects labels to be float and shape (N, 1)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    avg_loss = running_loss / dataset_size
    # Print full precision as requested
    print(f"Validation Loss: {avg_loss:.16f}")
    return avg_loss


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (Horizontal Flip).

    Args:
        model: PyTorch model.
        dataloader: Test DataLoader.
        device: Computation device.

    Returns:
        tuple: (probabilities, ids)
            probs: Flattened numpy array of probabilities.
            ids: Numpy array of image IDs.
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in dataloader:
            images = images.to(device)

            # 1. Prediction on original images
            logits = model(images)
            probs = torch.sigmoid(logits)

            # 2. Prediction on horizontally flipped images (TTA)
            if Config.USE_TTA:
                # Flip width dimension (N, C, H, W) -> dim 3
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average the probabilities
                probs = (probs + probs_flipped) / 2.0

            all_probs.append(probs.cpu().numpy().flatten())
            all_ids.extend(ids.numpy())

    return np.concatenate(all_probs), np.array(all_ids)


def train_model(model, train_loader, val_loader, device, fold_idx, backbone_name):
    """
    Orchestrates the training loop for a specific fold and backbone.
    Includes optimizer setup, scheduling, early stopping, and checkpointing.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Computation device.
        fold_idx: Integer index of the current fold.
        backbone_name: Name of the backbone (for file naming).

    Returns:
        tuple: (best_model, best_val_loss)
    """
    set_seed(Config.SEED)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    best_loss = float("inf")
    save_path = os.path.join(Config.WORKING_DIR, f"{backbone_name}_fold_{fold_idx}.pth")

    # Early Stopping settings
    patience = 3
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Backbone: {backbone_name} | Fold: {fold_idx} | Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.16f} | Val Loss: {val_loss:.16f}"
        )

        # Checkpointing and Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load the best model weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model, best_loss
