import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.model import SPPCNN
from library.utils import set_seed, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Unpack batch: images, angles, labels
        # IcebergDataset returns (img, angle, label) when y is provided
        images, angles, labels = batch
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch in loader:
            images, angles, labels = batch
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    val_loss = running_loss / dataset_size
    return val_loss


def train_fold(
    fold_idx,
    train_loader,
    val_loader,
    epochs=75,
    patience=12,
    lr=1e-3,
    weight_decay=1e-4,
    checkpoint_dir="./checkpoints",
):
    """
    Manages the training loop for a specific cross-validation fold.
    """
    device = get_device()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize Model
    model = SPPCNN().to(device)

    # Initialize Optimizer and Loss
    # Using AdamW with constant learning rate as per strategy
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, f"model_fold_{fold_idx}.pth")

    print(f"Starting training for Fold {fold_idx}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered for Fold {fold_idx} at epoch {epoch+1}")
            break

    print(f"Finished Fold {fold_idx}. Best Val Loss: {best_val_loss}")
    return best_val_loss


def predict(model, loader, device):
    """
    Generates predictions for a dataset.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle cases where loader returns (img, angle) or (img, angle, label)
            if len(batch) >= 2:
                images, angles = batch[0], batch[1]
            else:
                raise ValueError("Loader must return at least images and angles.")

            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())

    return np.array(preds)
