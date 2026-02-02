import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.models import get_model

# Constants
WORKING_DIR = "./working/idea_9"


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Lists to store predictions and labels for sklearn metric if needed,
    # but BCEWithLogitsLoss is mathematically equivalent to Log Loss.
    # We will use the accumulated loss for efficiency and consistency.

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def fit_model(
    model_name: str,
    resolution: int,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 3,
    num_workers: int = 4,
    debug_subset: int = None,
):
    """
    Orchestrates the training process for a specific model configuration.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"\nStarting training for {model_name} @ {resolution}x{resolution}...")

    # 1. Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        resolution=resolution,
        num_workers=num_workers,
        load_cached_data=True,
        debug_subset=debug_subset,
    )

    # 2. Model
    model = get_model(model_name=model_name, num_classes=1, pretrained=True)
    model = model.to(device)

    # 3. Optimizer & Scheduler & Loss
    # Using AdamW as per strategy
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-2
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # BCEWithLogitsLoss (No label smoothing as per strategy)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, f"{model_name}_best.pth")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Time: {elapsed:.1f}s - "
            f"Train Loss: {train_loss:.8f} - "
            f"Val Loss: {val_loss:.16f}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            # print(f"  EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished for {model_name}. Best Val Loss: {best_val_loss:.16f}")

    # Load best weights before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict(model, loader, device, use_tta=True):
    """
    Generates predictions for the test set.
    Implements Test Time Augmentation (Horizontal Flip).

    Returns:
        ids (list): List of image IDs.
        probs (np.array): Predicted probabilities (0-1).
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            # Unpack based on loader type (test loader returns image, id)
            images, img_ids = batch
            images = images.to(device)

            # 1. Forward Pass (Original)
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # 2. Forward Pass (Flipped)
                # Flip along width axis (dim 3 for NCHW)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(img_ids.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    return all_ids, all_probs
