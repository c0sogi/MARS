import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, FocalLoss, optimize_threshold
from library.data_processing import get_datasets
from library.model import SEARVN


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Unpack batch: (X_kin_cont, X_kin_cat, X_vis, y)
        x_kin_cont, x_kin_cat, x_vis, targets = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_kin_cont, x_kin_cat, x_vis)

        # Squeeze logits to match target shape if necessary (N, 1) -> (N,)
        logits = logits.squeeze()

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and optimizes threshold to find best MCC.
    """
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x_kin_cont, x_kin_cat, x_vis, targets = [b.to(device) for b in batch]

            logits = model(x_kin_cont, x_kin_cat, x_vis)
            logits = logits.squeeze()

            loss = criterion(logits, targets)
            running_loss += loss.item() * targets.size(0)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Aggregate results
    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Calculate average loss
    val_loss = running_loss / len(dataloader.dataset)

    # Calculate MCC
    # First convert logits to probabilities
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    # Optimize threshold for MCC
    best_thresh, best_mcc = optimize_threshold(
        all_targets, all_probs, steps=Config.THRESHOLD_SEARCH_STEPS
    )

    return val_loss, best_mcc, best_thresh


def run_training():
    """
    Orchestrates the training pipeline:
    1. Load Data
    2. Init Model/Optimizer/Loss
    3. Training Loop with Early Stopping
    4. Save Best Model
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading datasets...")
    train_ds, val_ds = get_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    print("Initializing SEA-RVN model...")
    model = SEARVN().to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # 5. Training Loop
    best_mcc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mcc, val_thresh = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val MCC: {val_mcc}")
        print(f"Best Threshold: {val_thresh}")

        # Early Stopping Check
        if val_mcc > best_mcc:
            print(f"MCC improved from {best_mcc} to {val_mcc}. Saving model...")
            best_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)
            # Also save the threshold
            np.save(os.path.join(Config.WORKING_DIR, "best_threshold.npy"), val_thresh)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # Load best model before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model
