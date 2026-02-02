import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.dataset import ContrailDataset
from library.model import TemporalAshNet
from library.loss import WeightedLoss


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# Training Engine
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0

    # Iterate over batches
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        # WeightedLoss returns (total, focal, dice)
        loss, focal_l, dice_l = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        running_focal += focal_l.item()
        running_dice += dice_l.item()

    # Average metrics
    n_batches = len(loader)
    avg_loss = running_loss / n_batches
    avg_focal = running_focal / n_batches
    avg_dice = running_dice / n_batches

    return avg_loss, avg_focal, avg_dice


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates Global Dice Coefficient and average Loss.
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for Global Dice
    # Dice = 2 * |X n Y| / (|X| + |Y|)
    global_intersection = 0.0
    global_union = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)

            # 1. Calculate Validation Loss
            loss, _, _ = criterion(logits, masks)
            running_loss += loss.item()

            # 2. Calculate Global Dice Metric
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # Flatten for calculation
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            global_intersection += intersection
            global_union += union

    # Average Loss
    avg_loss = running_loss / len(loader)

    # Global Dice Calculation
    # Add epsilon to avoid division by zero
    epsilon = 1e-6
    global_dice = (2.0 * global_intersection) / (global_union + epsilon)

    return avg_loss, global_dice


# ------------------------------------------------------------------------------
# Main Driver
# ------------------------------------------------------------------------------
def run_training(debug=False, early_stopping_patience=10):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Preparation
    print("Initializing Datasets...")
    train_dataset = ContrailDataset(Config.TRAIN_METADATA_PATH, stage="train")
    val_dataset = ContrailDataset(Config.VAL_METADATA_PATH, stage="validation")

    # Debugging: Slice datasets
    if debug:
        print(f"DEBUG MODE: Using {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_dataset.df = train_dataset.df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_dataset.df = val_dataset.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

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

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 3. Model, Loss, Optimizer
    print("Initializing Model...")
    model = TemporalAshNet().to(device)

    criterion = WeightedLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Reduce LR when Validation Dice plateaus
    # mode='max' because higher Dice is better
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE,
        min_lr=Config.LR_MIN,
        verbose=True,
    )

    # 4. Training Loop
    best_dice = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting Training...")
    print("-" * 80)

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # --- Train ---
        t_loss, t_focal, t_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # --- Validate ---
        v_loss, v_dice = validate(model, val_loader, criterion, device)

        # --- Scheduler Step ---
        # Step based on Validation Dice
        scheduler.step(v_dice)

        elapsed = time.time() - start_time

        # --- Logging ---
        print(f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.1f}s")
        print(
            f"    Train Loss: {t_loss:.6f} (Focal: {t_focal:.6f}, Dice: {t_dice:.6f})"
        )
        print(f"    Val Loss:   {v_loss:.6f}")
        print(f"    Val Dice:   {v_dice:.18f}")  # Full precision as requested

        # --- Checkpointing & Early Stopping ---
        if v_dice > best_dice:
            print(
                f"    (+) Validation Dice improved from {best_dice:.6f} to {v_dice:.6f}. Saving model."
            )
            best_dice = v_dice
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"    (.) Validation Dice did not improve. Patience: {patience_counter}/{early_stopping_patience}"
            )

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

        print("-" * 80)

    print(f"\nTraining Complete. Best Validation Dice: {best_dice:.18f}")
    print(f"Best model saved to: {best_model_path}")
