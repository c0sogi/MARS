import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import VolcanoDataset
from library.model import HybridModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for spec, tabular, target, _ in dataloader:
        spec = spec.to(device)
        tabular = tabular.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(spec, tabular)

        # Compute loss (MAE on scaled target)
        loss = criterion(outputs, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * target.size(0)
        count += target.size(0)

    epoch_loss = running_loss / count
    return epoch_loss


def validate(model, dataloader, criterion, device, scaler=None):
    """
    Evaluates the model on the validation set.
    Returns the scaled loss (for optimization) and original scale MAE (for reporting).
    """
    model.eval()
    running_loss = 0.0
    running_mae_original = 0.0
    count = 0

    with torch.no_grad():
        for spec, tabular, target, _ in dataloader:
            spec = spec.to(device)
            tabular = tabular.to(device)
            target = target.to(device)

            outputs = model(spec, tabular)

            # Loss on scaled target
            loss = criterion(outputs, target)
            running_loss += loss.item() * target.size(0)

            # Calculate MAE on original scale if scaler is provided
            if scaler:
                outputs_np = outputs.cpu().numpy()
                target_np = target.cpu().numpy()

                outputs_orig = scaler.inverse_transform(outputs_np)
                target_orig = scaler.inverse_transform(target_np)

                mae_orig = np.abs(outputs_orig - target_orig).sum()
                running_mae_original += mae_orig

            count += target.size(0)

    epoch_loss = running_loss / count
    epoch_mae_original = running_mae_original / count if scaler else 0.0

    return epoch_loss, epoch_mae_original


def generate_submission(device):
    """
    Generates predictions for the test set using the best saved model.
    """
    print("Starting submission generation...")

    # 1. Load Test Data
    test_dataset = VolcanoDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 2. Determine Input Dimension
    # Fetch one sample to determine tabular feature dimension
    _, sample_features, _, _ = test_dataset[0]
    tabular_input_dim = sample_features.shape[0]

    # 3. Load Model
    model = HybridModel(tabular_input_dim=tabular_input_dim)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Load Scaler for Inverse Transform
    scaler = TargetScaler()
    scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)

    # 5. Inference Loop
    predictions = []
    segment_ids = []

    with torch.no_grad():
        for spec, tabular, _, seg_ids in test_loader:
            spec = spec.to(device)
            tabular = tabular.to(device)

            # Predict (Scaled)
            outputs = model(spec, tabular)

            # Move to CPU
            outputs_np = outputs.cpu().numpy()

            # Inverse Scale
            outputs_orig = scaler.inverse_transform(outputs_np)

            predictions.extend(outputs_orig.tolist())
            segment_ids.extend(seg_ids.tolist())

    # 6. Save Submission
    df_sub = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main function to execute the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device selected: {device}")

    # 2. Data Loading
    print("Initializing datasets...")
    train_dataset = VolcanoDataset(mode="train")
    val_dataset = VolcanoDataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Determine tabular input dimension
    _, sample_features, _, _ = train_dataset[0]
    tabular_input_dim = sample_features.shape[0]
    print(f"Tabular Input Dimension: {tabular_input_dim}")

    # 3. Model Initialization
    model = HybridModel(tabular_input_dim=tabular_input_dim)
    model.to(device)

    # 4. Optimizer & Scheduler
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=False,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    scaler = train_dataset.scaler

    print("Starting training loop...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mae_orig = validate(model, val_loader, criterion, device, scaler)

        # Scheduler Update
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | LR: {current_lr} | "
            f"Train Loss (Scaled): {train_loss} | "
            f"Val Loss (Scaled): {val_loss} | "
            f"Val MAE (Original): {val_mae_orig}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved at epoch {epoch+1} with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Scaled Val Loss: {best_val_loss}")

    # 6. Generate Submission
    generate_submission(device)
