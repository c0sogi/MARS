import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Import provided library modules
from library.config import Config
from library.model import WSDN_ABS
from library.dataset import InkDataset, seed_everything
from library.loss import JointLoss
from library.utils import f05_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (volumes, targets) in enumerate(dataloader):
        volumes = volumes.to(device)

        # Move targets to device
        target_mask = targets["mask"].to(device)
        target_boundary = targets["boundary"].to(device)

        # Forward pass
        outputs = model(volumes)

        # Calculate loss (JointLoss expects dicts)
        loss_targets = {"mask": target_mask, "boundary": target_boundary}
        loss = criterion(outputs, loss_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Validation loop. Calculates Loss and F0.5 Score.
    """
    model.eval()
    running_loss = 0.0
    running_f05 = 0.0

    with torch.no_grad():
        for batch_idx, (volumes, targets) in enumerate(dataloader):
            volumes = volumes.to(device)

            target_mask = targets["mask"].to(device)
            target_boundary = targets["boundary"].to(device)

            outputs = model(volumes)

            # Calculate Loss
            loss_targets = {"mask": target_mask, "boundary": target_boundary}
            loss = criterion(outputs, loss_targets)
            running_loss += loss.item()

            # Calculate F0.5 Score for the Mask Head
            # Apply sigmoid to logits for probability
            mask_probs = torch.sigmoid(outputs["mask"])
            batch_f05 = f05_score(mask_probs, target_mask, threshold=0.5)
            running_f05 += batch_f05

    avg_loss = running_loss / len(dataloader)
    avg_f05 = running_f05 / len(dataloader)

    return avg_loss, avg_f05


def train():
    """
    Main training function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")
    Config.print_config()

    # 2. Data
    # Load cached data is True by default in dataset, but we can be explicit
    train_dataset = InkDataset(split="train", load_cached_data=True)
    val_dataset = InkDataset(split="val", load_cached_data=True)

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

    # 3. Model & Optimization
    model = WSDN_ABS(
        in_channels=Config.Z_DIM,
        model_channels=Config.MODEL_CHANNELS,
        dilation_rates=Config.DILATION_RATES,
    )
    model = model.to(device)

    criterion = JointLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_val_f05 = -1.0
    patience_counter = 0
    best_model_path = Config.WORKING_DIR / "best_model.pth"

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss}")

        # Validate
        val_loss, val_f05 = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val F0.5: {val_f05}")

        # Checkpointing & Early Stopping
        if val_f05 > best_val_f05:
            print(
                f"Validation F0.5 improved from {best_val_f05} to {val_f05}. Saving model..."
            )
            best_val_f05 = val_f05
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val F0.5: {best_val_f05}")
