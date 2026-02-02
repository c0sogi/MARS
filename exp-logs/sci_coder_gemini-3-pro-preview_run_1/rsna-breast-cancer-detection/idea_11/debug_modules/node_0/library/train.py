import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.model import PyramidSiameseEfficientNet
from library.data import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    # Iterate over batches
    # Loader returns: ((tensor_target, tensor_contra), label)
    for batch_idx, (inputs, targets) in enumerate(loader):
        # Unpack inputs
        img_target, img_contra = inputs

        # Move to device
        img_target = img_target.to(device, non_blocking=True)
        img_contra = img_contra.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # The model expects (x_target, x_contra)
        logits = model(img_target, img_contra)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Update weights
        # Note: Gradient clipping is explicitly DISABLED as per strategy
        # to allow large updates for the minority class given the high pos_weight.
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            img_target, img_contra = inputs

            img_target = img_target.to(device, non_blocking=True)
            img_contra = img_contra.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True).unsqueeze(1)

            logits = model(img_target, img_contra)
            loss = criterion(logits, targets_dev)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # Calculate pF1
        pf1 = probabilistic_f1(all_targets, all_probs)
    else:
        pf1 = 0.0

    return avg_loss, pf1


def run_training(load_cached_data=True):
    """
    Main execution function for training.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    print("Initializing Model...")
    model = PyramidSiameseEfficientNet()
    model.to(device)

    # 4. Loss Function
    # Aggressive positive weighting to approximate inverse class frequency
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 6. Training Loop
    best_pf1 = -1.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss:.10f}")
        print(f"Val Loss:   {val_loss:.10f}")
        print(f"Val pF1:    {val_pf1:.10f}")

        # Save Best Model
        if val_pf1 > best_pf1:
            print(f"New Best pF1! ({best_pf1:.10f} -> {val_pf1:.10f}). Saving model...")
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            print(f"pF1 did not improve (Best: {best_pf1:.10f})")

        print("-" * 50)

    print("Training complete.")
    print(f"Best Validation pF1: {best_pf1:.10f}")
    print(f"Model saved to: {Config.MODEL_PATH}")
