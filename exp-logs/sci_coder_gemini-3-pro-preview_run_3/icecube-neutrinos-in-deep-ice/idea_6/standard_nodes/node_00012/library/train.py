import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.data import get_dataloaders
from library.model import CFDGN, criterion, compute_mae
from library import utils


def set_seed(seed):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()

        # Forward pass
        pred = model(x, mask)

        # Compute loss
        loss = criterion(pred, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * batch_size
        running_mae += compute_mae(pred, target) * batch_size
        total_samples += batch_size

    return running_loss / total_samples, running_mae / total_samples


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            batch_size = x.size(0)

            pred = model(x, mask)
            loss = criterion(pred, target)

            running_loss += loss.item() * batch_size
            running_mae += compute_mae(pred, target) * batch_size
            total_samples += batch_size

    return running_loss / total_samples, running_mae / total_samples


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set, handling the inverse rotation
    from Canonical Frame to Global Frame.
    """
    model.eval()
    results = []
    print("Generating predictions...")

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            rotation = batch["rotation"].numpy()  # (B, 3, 3)
            event_ids = batch["event_id"].numpy()

            # Predict direction in Canonical Frame: (B, 3)
            pred_local = model(x, mask).cpu().numpy()

            # Inverse Rotation: Transform back to Global Frame
            # v_global = R^T @ v_local
            # Using einsum for batch matrix multiplication:
            # rotation is (B, j, i) where j is local, i is global (since R aligns global to local)
            # Actually, R aligns Principal Axis (local) to Z (global target in utils logic? No.)
            # Utils logic: R rotates Principal Axis TO Target Axis (0,0,1).
            # So R maps Global -> Local (Canonical).
            # Therefore, Inverse is R^T.
            # v_global = R^T @ v_local
            # einsum "bji,bj->bi":
            # b: batch
            # j: local dimension (summed over)
            # i: global dimension (output)
            # R[b, j, i] is the element at row j, col i of R.
            # If R is (3,3), R[row, col].
            # We want v_g = R.T @ v_l.
            # v_g[i] = sum_j (R.T[i, j] * v_l[j]) = sum_j (R[j, i] * v_l[j])
            pred_global = np.einsum("bji,bj->bi", rotation, pred_local)

            # Convert Cartesian to Spherical (Azimuth, Zenith)
            az, ze = utils.cartesian_to_spherical(
                pred_global[:, 0], pred_global[:, 1], pred_global[:, 2]
            )

            # Collect results
            for eid, a, z in zip(event_ids, az, ze):
                results.append({"event_id": int(eid), "azimuth": a, "zenith": z})

    # Create DataFrame
    df_sub = pd.DataFrame(results)

    # Ensure correct column order
    df_sub = df_sub[["event_id", "azimuth", "zenith"]]

    # Save to CSV
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_sub)} predictions.")


def run():
    """
    Main execution pipeline: Setup -> Train -> Validate -> Inference.
    """
    # 1. Setup
    Config.setup_directories()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Initializing CF-DGN on {device}...")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    model = CFDGN().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_val_mae = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train & Validate
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        duration = time.time() - start_time

        # Log Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {duration:.2f}s | "
            f"Train Loss: {train_loss} | Train MAE: {train_mae} | "
            f"Val Loss: {val_loss} | Val MAE: {val_mae}"
        )

        # Checkpointing & Early Stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("  -> Model Saved (New Best)")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Inference
    print(f"Loading best model from {Config.MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
