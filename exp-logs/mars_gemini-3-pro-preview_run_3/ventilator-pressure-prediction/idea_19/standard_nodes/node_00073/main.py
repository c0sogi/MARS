import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device, MaskedL1Loss
from library.dataset import get_data_loaders, add_features
from library.model import WSDHNet, predict
from library.train import train_epoch, validate


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 6
    Config.BATCH_SIZE = 512  # Increased for A100 speed
    Config.NUM_WORKERS = 8  # Utilize available vCPUs

    Config.initialize()
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading and processing data...")
    # load_cached_data=True allows using pre-computed parquet files if available
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # Determine input dimension dynamically
    sample_x, _, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2]
    print(f"Detected Input Dimension: {input_dim}")

    # ==========================================
    # 3. Model & Optimizer Setup
    # ==========================================
    model = WSDHNet(input_dim=input_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    criterion = MaskedL1Loss()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_val_loss = float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! (Val Loss: {best_val_loss:.6f})")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")

    # ==========================================
    # 5. Final Validation & Metric Calculation
    # ==========================================
    print("\nRunning full validation on best model...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []
    all_u_out = []

    # Inference on Validation Set (No Gradients)
    with torch.no_grad():
        for x, u_out, y in val_loader:
            x = x.to(device)
            u_out = u_out.to(device)
            y = y.to(device)

            preds = model(x)

            # Move to CPU for analysis to save GPU memory
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_inputs.append(x.cpu())
            all_u_out.append(u_out.cpu())

    # Concatenate all batches
    preds_tensor = torch.cat(all_preds)
    targets_tensor = torch.cat(all_targets)
    inputs_tensor = torch.cat(all_inputs)
    u_out_tensor = torch.cat(all_u_out)

    # Calculate Metric: MAE on Inspiratory Phase (u_out == 0)
    # Using < 0.5 to be robust against float precision, though == 0 is standard here
    mask = u_out_tensor < 0.5

    masked_preds = preds_tensor[mask]
    masked_targets = targets_tensor[mask]

    final_metric = torch.abs(masked_preds - masked_targets).mean().item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")

    # Calculate element-wise absolute errors for the inspiratory phase
    errors = torch.abs(preds_tensor - targets_tensor)
    masked_errors = errors[mask].numpy()
    masked_inputs = inputs_tensor[mask].numpy()

    # Attempt to retrieve feature names for better reporting
    try:
        # Load a tiny chunk of raw data to reconstruct feature columns
        dummy_df = pd.read_csv(Config.TRAIN_CSV, nrows=10)
        dummy_df = add_features(dummy_df)

        exclude_cols = ["id", "breath_id", "pressure", "u_out"]
        if Config.EXCLUDE_RAW_TIME:
            exclude_cols.append("time_step")

        feature_names = [c for c in dummy_df.columns if c not in exclude_cols]
        # Verify length matches
        if len(feature_names) != input_dim:
            feature_names = [f"Feature_{i}" for i in range(input_dim)]
    except Exception as e:
        print(f"Warning: Could not infer feature names ({e}). Using indices.")
        feature_names = [f"Feature_{i}" for i in range(input_dim)]

    # Calculate correlation between Error Magnitude and Feature Value
    correlations = []
    for i in range(input_dim):
        feat_vals = masked_inputs[:, i]
        # Avoid correlation with constant features
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(masked_errors, feat_vals)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Correlations between Error and Input Features:")
    for name, corr in correlations[:10]:
        print(f"{name:<20}: {corr:.4f}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 0.1642141044139862

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        # predict function handles loading model, inference, and saving CSV
        predict(model, test_loader, device)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
