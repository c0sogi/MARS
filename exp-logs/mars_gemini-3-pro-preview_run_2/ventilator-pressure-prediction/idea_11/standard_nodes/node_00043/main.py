import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Ensure current directory is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders, compute_features
from library.model import DP_GI_BiLSTM
from library.train import train_epoch, validate, get_u_out_metadata, predict
from library.inference import create_submission


def get_feature_names():
    """
    Reconstructs the feature column names by simulating the feature engineering pipeline.
    This is necessary to map feature indices to names for failure analysis.
    """
    cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    # Create dummy data
    df_dummy = pd.DataFrame(np.zeros((2, len(cols))), columns=cols)
    df_dummy["R"] = 20
    df_dummy["C"] = 10
    df_dummy["breath_id"] = 1
    df_dummy["time_step"] = [0.0, 0.1]

    # Apply pipeline
    df_processed = compute_features(df_dummy)

    # Replicate sorting logic from dataset.py
    exclude_cols = ["id", "breath_id", "pressure"]
    feature_cols = sorted([c for c in df_processed.columns if c not in exclude_cols])
    return feature_cols


def run_failure_analysis(
    model, val_loader, device, u_out_idx, u_out_mean, u_out_std, feature_names
):
    """
    Calculates and prints the correlation between absolute error and input features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_errors = []
    all_features = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            # Unscale u_out to identify inspiratory phase
            u_out_scaled = X[:, :, u_out_idx]
            u_out_raw = (u_out_scaled * u_out_std) + u_out_mean

            # Mask: Inspiratory phase only (u_out approx 0)
            mask = u_out_raw < 0.5

            # Calculate Absolute Error
            abs_error = torch.abs(pred - y)

            # Filter data using mask
            valid_error = abs_error[mask]
            valid_features = X[mask]  # Shape: (N_samples, N_features)

            if len(valid_error) > 0:
                all_errors.append(valid_error.cpu().numpy())
                all_features.append(valid_features.cpu().numpy())

    # Concatenate all batches
    if not all_errors:
        print("No inspiratory phase data found for analysis.")
        return

    all_errors = np.concatenate(all_errors)
    all_features = np.concatenate(all_features, axis=0)

    print(f"Analyzed {len(all_errors)} inspiratory time steps.")
    print("Correlation between Absolute Error and Input Features:")

    correlations = {}
    for i, name in enumerate(feature_names):
        feat_vals = all_features[:, i]
        # Calculate Pearson correlation
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(all_errors, feat_vals)[0, 1]
        correlations[name] = corr

    # Sort by absolute correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for name, corr in sorted_corr:
        print(f"{name}: {corr:.4f}")
    print("========================\n")


def main():
    # 1. Configuration
    # Extending epochs to ensure deep convergence (Cite solution_lesson_node_00029)
    Config.EPOCHS = 180

    # 2. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 3. Data Loading
    print("Loading data...")
    # This updates Config.INPUT_DIM internally
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Retrieve metadata for u_out (needed for validation masking)
    u_out_idx, u_out_mean, u_out_std = get_u_out_metadata(Config.CACHE_DIR)
    feature_names = get_feature_names()

    # 4. Model Initialization
    print(f"Initializing DP-GI-BiLSTM with Input Dim: {Config.INPUT_DIM}...")
    model = DP_GI_BiLSTM(Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_mae = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, u_out_idx, u_out_mean, u_out_std
        )

        val_mae = validate(model, val_loader, device, u_out_idx, u_out_mean, u_out_std)

        scheduler.step()

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)

    print("Training complete.")

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    final_val_mae = validate(
        model, val_loader, device, u_out_idx, u_out_mean, u_out_std
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_mae}")

    # 7. Failure Analysis
    run_failure_analysis(
        model, val_loader, device, u_out_idx, u_out_mean, u_out_std, feature_names
    )

    # 8. Conditional Submission
    THRESHOLD = 0.19168047735075858

    if final_val_mae < THRESHOLD:
        print(
            f"Metric ({final_val_mae}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        preds = predict(model, test_loader, device)
        create_submission(preds, test_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_val_mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
