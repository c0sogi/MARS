import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.data import get_dataloaders
from library.model import FractureMILModel
from library.utils import calculate_weighted_loss_metric, seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class FractureLoss(nn.Module):
    """
    Implicitly Weighted Multi-Task Loss.
    L = mean(BCE_C1..C7) + BCE_Patient

    This formulation creates a 1:7 weighting ratio between individual vertebrae
    and the patient outcome, matching the competition metric's intent without
    explicit scalar multipliers that might distort gradient magnitudes.
    """

    def __init__(self):
        super().__init__()
        # BCEWithLogitsLoss combines Sigmoid and BCE.
        # reduction='mean' calculates the mean over the batch and spatial dimensions.
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets):
        # logits, targets: (Batch, 8)
        # Columns 0-6: C1-C7
        # Column 7: patient_overall

        # 1. Vertebrae Loss (Mean over C1-C7)
        # We slice [:, :7] to get the 7 vertebrae.
        # The BCE function with reduction='mean' will average over the batch AND the 7 classes.
        loss_vertebrae = self.bce(logits[:, :7], targets[:, :7])

        # 2. Patient Overall Loss
        # We slice [:, 7] to get the patient outcome.
        loss_patient = self.bce(logits[:, 7], targets[:, 7])

        # 3. Total Loss
        return loss_vertebrae + loss_patient


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    """
    Trains the model for one epoch using Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Mixed Precision Context
        with autocast():
            # Forward pass
            # images shape: (B, Slices, 3, H, W)
            logits = model(images)

            # Calculate loss
            loss = criterion(logits, labels)

        # Backward pass with Scaler
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model and calculates the official metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Store predictions and targets for metric calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in enumerate(loader):
            # Unpack tuple (idx, (images, labels)) or just (images, labels) depending on loader
            if isinstance(images, int):
                # If enumerate was used on loader directly without unpacking in loop definition
                # But here we used `for images, labels in loader` usually, but let's be safe
                pass

            # Correct loop unpacking for DataLoader
            pass

    # Re-writing loop for clarity/correctness
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.float().cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate official metric
    metric_score = calculate_weighted_loss_metric(all_targets, all_preds)

    return epoch_loss, metric_score


def run_training(debug=False):
    """
    Main training function.
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug or Config.DEBUG:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # 3. Model
    print(f"Initializing model: {Config.BACKBONE}")
    model = FractureMILModel(config=Config)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Gradient Scaler for AMP
    scaler = GradScaler()

    # Decoupled Cosine Annealing
    # T_max set to 1.5x epochs to avoid restarting and allow full decay curve
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Config.EPOCHS * Config.T_MAX_MULT), eta_min=Config.MIN_LR
    )

    criterion = FractureLoss()

    # 5. Training Loop
    best_loss = float("inf")
    best_metric = float("inf")

    # Early stopping parameters
    patience = 3
    counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )

        # Validate
        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Metric (Weighted LogLoss): {val_metric}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Checkpointing based on Validation Loss (primary objective for stability)
        if val_loss < best_loss:
            print(
                f"Validation Loss improved from {best_loss} to {val_loss}. Saving model..."
            )
            best_loss = val_loss
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            counter = 0
        else:
            counter += 1
            print(f"Validation Loss did not improve. Counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Validation Loss: {best_loss}")
    print(f"Best Validation Metric: {best_metric}")
