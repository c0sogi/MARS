import torch
import pandas as pd
import numpy as np
import os
import sys

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.utils import get_device
from library.dataset import DataManager
from library.model import DisentangledTCNLSTM
from library.engine import train_fn, eval_fn


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Configure for Fast Baseline Execution
    # We override Config parameters to limit training time while maintaining validity.
    Config.EPOCHS = 10
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 15000  # Train on 15,000 breaths for speed

    # 3. Data Loading
    dm = DataManager()

    # Train Loader: Use Debug mode to limit samples
    print("Initializing Training Loader (Debug Mode)...")
    train_loader = dm.get_dataloader("train", shuffle=True, load_cached_data=True)

    # Val Loader: Use Full mode (disable debug temporarily)
    # Requirement: "print the final validation metric computed on the entire hold-out validation set"
    Config.DEBUG = False
    print("Initializing Validation Loader (Full Set)...")
    val_loader = dm.get_dataloader("validation", shuffle=False, load_cached_data=True)

    # 4. Model Initialization
    model = DisentangledTCNLSTM().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Training Loop
    best_val_loss = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, device)
        val_loss = eval_fn(model, val_loader, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> Saved best model with Loss: {best_val_loss:.6f}")

    # 6. Final Validation Metric
    print(f"Final Validation Metric: {best_val_loss}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load best model for analysis
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    errors = []
    feats_u_in = []
    feats_R = []
    feats_C = []

    # Identify feature indices from Config
    # SKIP_FEATURES = ["R", "C", "R_flow", "C_volume", "u_out"]
    # TCN_FEATURES = ["u_in", "u_in_diff1", "u_in_diff2"]
    try:
        u_out_idx = Config.SKIP_FEATURES.index("u_out")
        r_idx = Config.SKIP_FEATURES.index("R")
        c_idx = Config.SKIP_FEATURES.index("C")
        u_in_idx = Config.TCN_FEATURES.index("u_in")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            preds = model(inputs)

            # Extract u_out for masking (we analyze inspiratory phase errors)
            u_out = inputs["skip"][:, :, u_out_idx]

            # Calculate Absolute Error
            abs_error = torch.abs(preds - targets)

            # Mask: Only consider inspiratory phase (u_out == 0)
            mask = u_out == 0

            # Filter valid elements
            valid_error = abs_error[mask]

            # Extract features corresponding to valid elements
            # inputs['tcn'] is (B, C, L). u_in is channel 0. -> (B, L)
            u_in = inputs["tcn"][:, u_in_idx, :]
            valid_u_in = u_in[mask]

            # inputs['skip'] is (B, L, C).
            R = inputs["skip"][:, :, r_idx]
            valid_R = R[mask]

            C = inputs["skip"][:, :, c_idx]
            valid_C = C[mask]

            # Move to CPU and store
            errors.append(valid_error.cpu().numpy())
            feats_u_in.append(valid_u_in.cpu().numpy())
            feats_R.append(valid_R.cpu().numpy())
            feats_C.append(valid_C.cpu().numpy())

    # Concatenate all batches
    all_errors = np.concatenate(errors)
    all_u_in = np.concatenate(feats_u_in)
    all_R = np.concatenate(feats_R)
    all_C = np.concatenate(feats_C)

    # Create DataFrame for correlation calculation
    df_analysis = pd.DataFrame(
        {"error": all_errors, "u_in": all_u_in, "R": all_R, "C": all_C}
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 8. Submission Logic
    THRESHOLD = 0.8097341656684875

    if best_val_loss < THRESHOLD:
        print(f"\nMetric {best_val_loss} < {THRESHOLD}. Generating submission...")

        # Load Test Data (Full)
        test_loader = dm.get_dataloader("test", shuffle=False, load_cached_data=True)

        all_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                preds = model(inputs)
                all_preds.append(preds.cpu().numpy())

        # Flatten predictions (N_breaths * 80)
        flat_preds = np.concatenate(all_preds, axis=0).flatten()

        # Load Test Metadata for IDs and alignment
        print("Aligning predictions with test IDs...")
        test_df = pd.read_csv(Config.TEST_PATH)
        # Ensure sorting matches DataManager logic (breath_id, time_step)
        test_df.sort_values(["breath_id", "time_step"], inplace=True)

        submission = pd.DataFrame({"id": test_df["id"], "pressure": flat_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {best_val_loss} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
