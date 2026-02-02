import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import FFDCNet
from library.utils import seed_everything, find_best_threshold


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch_idx, (volumes, labels) in enumerate(loader):
        volumes = volumes.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(volumes)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accumulate batch loss (multiply by batch size)
        running_loss += loss.item() * volumes.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set and finds the best F0.5 threshold.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(volumes)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * volumes.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Move to CPU and flatten for metric calculation
            all_preds.append(probs.cpu().numpy().flatten())
            all_targets.append(labels.cpu().numpy().flatten())

    avg_loss = running_loss / dataset_size

    # Concatenate all batches into single arrays
    y_pred_probs = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Find the threshold that maximizes F0.5 score
    best_threshold, best_score = find_best_threshold(y_true, y_pred_probs, beta=0.5)

    return avg_loss, best_score, best_threshold


def train_model(load_cached_data=True):
    """
    Main function to setup data, model, and run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    train_dataset = InkDataset(Config.TRAIN_METADATA, load_cached_data=load_cached_data)
    val_dataset = InkDataset(Config.VAL_METADATA, load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = FFDCNet().to(device)

    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Handle class imbalance with pos_weight
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Training Loop
    best_f05 = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    threshold_path = os.path.join(Config.WORKING_DIR, "best_threshold.txt")

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_f05, val_thresh = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics (full precision)
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Time: {elapsed}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F0.5: {val_f05}")
        print(f"Best Threshold: {val_thresh}")

        # Checkpointing & Early Stopping
        if val_f05 > best_f05:
            best_f05 = val_f05
            patience_counter = 0

            # Save Model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")

            # Save Threshold
            with open(threshold_path, "w") as f:
                f.write(str(val_thresh))
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model, best_f05
