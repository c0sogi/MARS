import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    DEVICE,
    SEED,
    NUM_WORKERS,
    seed_everything,
)
from library.model import TemporalCNN
from library.data_loader import get_dataloader
from library.utils import direction_to_angles, angular_dist_score


def train_epoch(model, dataloader, optimizer, device):
    """
    Performs one training epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs)

        # Loss: 1 - Cosine Similarity
        # preds and targets are (B, 3)
        cos_sim = F.cosine_similarity(preds, targets, dim=1)
        loss = 1.0 - cos_sim.mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Mean Angular Error.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds_az = []
    all_preds_zen = []
    all_true_az = []
    all_true_zen = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(inputs)

            # Loss calculation
            cos_sim = F.cosine_similarity(preds, targets, dim=1)
            loss = 1.0 - cos_sim.mean()
            running_loss += loss.item()
            num_batches += 1

            # Metric Calculation
            # Normalize predictions to unit vectors
            preds_norm = F.normalize(preds, p=2, dim=1)

            # Convert to angles
            pred_az, pred_zen = direction_to_angles(
                preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2]
            )
            true_az, true_zen = direction_to_angles(
                targets[:, 0], targets[:, 1], targets[:, 2]
            )

            # Collect for full metric calculation
            all_preds_az.append(pred_az.cpu().numpy())
            all_preds_zen.append(pred_zen.cpu().numpy())
            all_true_az.append(true_az.cpu().numpy())
            all_true_zen.append(true_zen.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches to compute global metric
    if len(all_preds_az) > 0:
        y_pred_az = np.concatenate(all_preds_az)
        y_pred_zen = np.concatenate(all_preds_zen)
        y_true_az = np.concatenate(all_true_az)
        y_true_zen = np.concatenate(all_true_zen)

        y_pred = np.stack([y_pred_az, y_pred_zen], axis=1)
        y_true = np.stack([y_true_az, y_true_zen], axis=1)

        mae = angular_dist_score(y_true, y_pred)
    else:
        mae = 0.0

    return avg_loss, mae


def train_model(max_train_samples=None, max_val_samples=None, epochs=EPOCHS):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    # Reproducibility
    seed_everything(SEED)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    model_save_path = os.path.join(WORKING_DIR, "best_model.pth")

    # Initialize DataLoaders
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        TRAIN_META_PATH,
        mode="train",
        max_samples=max_train_samples,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    val_loader = get_dataloader(
        VAL_META_PATH,
        mode="val",
        max_samples=max_val_samples,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Initialize Model, Optimizer, Scheduler
    print(f"Initializing Model on {DEVICE}...")
    model = TemporalCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)

        # Validate
        val_loss, val_mae = validate(model, val_loader, DEVICE)

        # Scheduler Step
        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Time: {elapsed}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MAE: {val_mae}"
        )

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs.")
                break

    return model_save_path


def predict_and_submit(model_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Starting inference on Test Set...")

    # Load Model
    model = TemporalCNN().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.eval()

    # Test DataLoader
    test_loader = get_dataloader(
        TEST_META_PATH,
        mode="test",
        batch_size=BATCH_SIZE * 2,  # Can use larger batch size for inference
        num_workers=NUM_WORKERS,
    )

    all_event_ids = []
    all_azimuths = []
    all_zeniths = []

    with torch.no_grad():
        for inputs, event_ids in test_loader:
            inputs = inputs.to(DEVICE)

            # Forward pass
            preds = model(inputs)

            # Normalize to unit vectors
            preds_norm = F.normalize(preds, p=2, dim=1)

            # Convert to angles
            az, zen = direction_to_angles(
                preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2]
            )

            # Store results
            all_event_ids.extend(event_ids.numpy())
            all_azimuths.extend(az.cpu().numpy())
            all_zeniths.extend(zen.cpu().numpy())

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"event_id": all_event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
    )

    # Sort by event_id
    submission_df.sort_values("event_id", inplace=True)

    # Save to CSV
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path} with shape {submission_df.shape}")
