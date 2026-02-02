import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import get_dataloaders
from library.model import HybridModel


def train_one_epoch(dataloader, model, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for spec, tab, target in dataloader:
        spec = spec.to(device)
        tab = tab.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(spec, tab)

        # Loss calculation (on scaled targets)
        loss = criterion(preds, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * target.size(0)
        count += target.size(0)

    return running_loss / count


def validate(dataloader, model, criterion, target_scaler, device):
    """
    Evaluates the model on the validation set.
    Returns the loss (scaled) and the MAE (original scale).
    """
    model.eval()
    running_loss = 0.0
    total_mae = 0.0
    count = 0

    with torch.no_grad():
        for spec, tab, target in dataloader:
            spec = spec.to(device)
            tab = tab.to(device)
            target = target.to(device)

            preds = model(spec, tab)
            loss = criterion(preds, target)

            running_loss += loss.item() * target.size(0)

            # Calculate MAE in original scale
            preds_unscaled = target_scaler.inverse_transform(preds.cpu().numpy())
            target_unscaled = target_scaler.inverse_transform(target.cpu().numpy())

            # Accumulate absolute error
            batch_mae = np.sum(np.abs(preds_unscaled - target_unscaled))
            total_mae += batch_mae

            count += target.size(0)

    avg_loss = running_loss / count
    avg_mae = total_mae / count

    return avg_loss, avg_mae


def generate_submission(dataloader, model, target_scaler, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()
    predictions = []

    # We need to map predictions back to segment_ids.
    # The dataloader iterates in the order of the underlying dataset.
    # We can retrieve the segment_ids from the dataset's metadata.

    with torch.no_grad():
        for spec, tab in dataloader:
            spec = spec.to(device)
            tab = tab.to(device)

            preds = model(spec, tab)

            # Inverse transform to get actual time_to_eruption
            preds_unscaled = target_scaler.inverse_transform(preds.cpu().numpy())
            predictions.extend(preds_unscaled)

    # Retrieve segment_ids from the dataset metadata
    # The dataset merges metadata with features, so the order is preserved
    test_meta = dataloader.dataset.data
    segment_ids = test_meta["segment_id"].values

    # Create DataFrame
    df_sub = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True):
    """
    Main orchestration function for training, validation, and submission.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    loaders = get_dataloaders(load_cached_data=load_cached_data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Determine input dimension for MLP branch
    # Get one batch to check shape
    # Dataset returns (spec, tab, target)
    sample_spec, sample_tab, _ = train_loader.dataset[0]
    num_tabular_features = sample_tab.shape[0]
    print(f"Detected {num_tabular_features} tabular features.")

    # Load Target Scaler for metric calculation
    target_scaler = TargetScaler()
    target_scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)

    # 3. Model Initialization
    model = HybridModel(num_tabular_features=num_tabular_features)
    model = model.to(device)

    # 4. Optimizer & Loss
    criterion = nn.L1Loss()  # MAE on scaled targets
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 5. Training Loop
    best_val_mae = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device)

        # Validate
        val_loss, val_mae = validate(
            val_loader, model, criterion, target_scaler, device
        )

        # Scheduler Step
        scheduler.step(val_mae)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss (Scaled MAE): {train_loss:.6f} | "
            f"Val Loss (Scaled MAE): {val_loss:.6f} | "
            f"Val MAE (Original): {val_mae}"
        )

        # Early Stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved! (MAE: {best_val_mae})")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    generate_submission(
        test_loader, model, target_scaler, device, Config.SUBMISSION_PATH
    )
