import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, get_device, MetricMonitor
from library.dataset import get_dataloaders, compute_features
from library.model import DP_GI_BiLSTM


def get_u_out_metadata(scaler_dir):
    """
    Determines the feature index of 'u_out' and loads its scaler statistics.
    This is required to unscale 'u_out' inside the loss function to identify
    inspiratory vs expiratory phases.
    """
    # 1. Determine Index by simulating the feature generation pipeline
    cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    # Create dummy data with valid values to prevent calculation errors
    df_dummy = pd.DataFrame(np.zeros((2, len(cols))), columns=cols)
    df_dummy["R"] = 20
    df_dummy["C"] = 10
    df_dummy["breath_id"] = 1
    df_dummy["time_step"] = [0.0, 0.1]

    # Apply feature engineering to get the exact column order
    df_processed = compute_features(df_dummy)

    # Replicate column sorting logic from dataset.py
    exclude_cols = ["id", "breath_id", "pressure"]
    feature_cols = sorted([c for c in df_processed.columns if c not in exclude_cols])

    try:
        u_out_idx = feature_cols.index("u_out")
    except ValueError:
        raise ValueError("'u_out' not found in feature columns.")

    # 2. Load Scaler Stats
    scaler_path = os.path.join(scaler_dir, "scaler_params.npz")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler params not found at {scaler_path}")

    scaler_data = np.load(scaler_path)
    mean = scaler_data["mean"]
    std = scaler_data["std"]

    u_out_mean = mean[u_out_idx]
    u_out_std = std[u_out_idx]

    return u_out_idx, u_out_mean, u_out_std


def train_epoch(model, loader, optimizer, device, u_out_idx, u_out_mean, u_out_std):
    """
    Executes one training epoch using Weighted L1 Loss.
    """
    model.train()
    monitor = MetricMonitor()

    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(X)

        # Extract u_out for weighting
        # X shape: (Batch, Seq, Feat)
        u_out_scaled = X[:, :, u_out_idx]

        # Unscale u_out to get binary mask (approx 0 or 1)
        u_out_raw = (u_out_scaled * u_out_std) + u_out_mean

        # Weighted L1 Loss
        # Inspiratory (u_out < 0.5): Weight 1.0
        # Expiratory (u_out > 0.5): Weight 0.1
        weights = torch.ones_like(u_out_raw) * Config.LOSS_WEIGHT_INSPIRATORY
        weights[u_out_raw > 0.5] = Config.LOSS_WEIGHT_EXPIRATORY

        loss = (torch.abs(pred - y) * weights).mean()

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        monitor.update("Loss", loss.item(), X.size(0))

    return monitor.metrics["Loss"]["avg"]


def validate(model, loader, device, u_out_idx, u_out_mean, u_out_std):
    """
    Evaluates the model using MAE on the Inspiratory Phase only.
    """
    model.eval()
    monitor = MetricMonitor()

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            # Extract and unscale u_out
            u_out_scaled = X[:, :, u_out_idx]
            u_out_raw = (u_out_scaled * u_out_std) + u_out_mean

            # Metric: MAE on Inspiratory Phase ONLY (u_out == 0)
            # Using 0.5 as safe threshold for binary float
            mask = u_out_raw < 0.5

            # Filter predictions and targets
            pred_insp = pred[mask]
            y_insp = y[mask]

            if len(pred_insp) > 0:
                mae = torch.abs(pred_insp - y_insp).mean()
                monitor.update("MAE", mae.item(), len(pred_insp))

    return monitor.metrics["MAE"]["avg"]


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy().flatten())

    return np.concatenate(predictions)


def run_training():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Preparing Dataloaders...")
    # This triggers feature engineering and scaler generation
    train_loader, val_loader, test_loader, test_ids = get_dataloaders()

    # 3. Retrieve Metadata
    # Must be done after get_dataloaders ensures scaler_params.npz exists
    print("Retrieving feature metadata...")
    u_out_idx, u_out_mean, u_out_std = get_u_out_metadata(Config.CACHE_DIR)

    # 4. Model Initialization
    print("Initializing Model...")
    # Config.INPUT_DIM is updated by get_dataloaders
    model = DP_GI_BiLSTM(Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_mae = float("inf")
    patience = 50
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, u_out_idx, u_out_mean, u_out_std
        )

        val_mae = validate(model, val_loader, device, u_out_idx, u_out_mean, u_out_std)

        # Step scheduler
        scheduler.step()

        print(f"Epoch {epoch} | Train Loss: {train_loss} | Val MAE: {val_mae}")

        # Checkpointing & Early Stopping
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print("  -> New Best Model Saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training Complete. Best Val MAE: {best_mae}")

    # 6. Submission Generation
    print("Generating Submission...")
    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    preds = predict(model, test_loader, device)

    # Ensure alignment
    if len(test_ids) != len(preds):
        print(f"Warning: Length mismatch. IDs: {len(test_ids)}, Preds: {len(preds)}")

    submission_df = pd.DataFrame({"id": test_ids, "pressure": preds})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_training()
