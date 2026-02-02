import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os

from library.config import Config
from library.data import CrystalDataset, collate_batch
from library.model import LCDS
from library.utils import inverse_log_transform, compute_rmsle


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move batch data to device
        atom_types = batch["atom_types"].to(device)
        dist_matrix = batch["dist_matrix"].to(device)
        lattice_features = batch["lattice_features"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atom_types, dist_matrix, lattice_features, mask)

        # Compute loss (MSE on log-transformed targets)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using RMSLE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            atom_types = batch["atom_types"].to(device)
            dist_matrix = batch["dist_matrix"].to(device)
            lattice_features = batch["lattice_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            outputs = model(atom_types, dist_matrix, lattice_features, mask)

            # Inverse transform to get original scale for RMSLE calculation
            # The dataset returns log(1+x), so we apply exp(x)-1
            preds_original = inverse_log_transform(outputs)
            targets_original = inverse_log_transform(targets)

            all_preds.append(preds_original.cpu().numpy())
            all_targets.append(targets_original.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute RMSLE
    score = compute_rmsle(all_targets, all_preds)
    return score


def train_model(num_epochs=Config.NUM_EPOCHS, limit_data=None):
    """
    Main function to train the model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize Datasets and Dataloaders
    train_dataset = CrystalDataset(mode="train", limit=limit_data)
    val_dataset = CrystalDataset(mode="val", limit=limit_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Initialize Model
    model = LCDS().to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Training Loop with Early Stopping
    best_val_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val RMSLE: {val_score:.6f}"
        )

        # Scheduler step
        scheduler.step(val_score)

        # Early Stopping and Checkpointing
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"  -> New best model saved! RMSLE: {best_val_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation RMSLE: {best_val_score:.6f}")

    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH))
    return model
