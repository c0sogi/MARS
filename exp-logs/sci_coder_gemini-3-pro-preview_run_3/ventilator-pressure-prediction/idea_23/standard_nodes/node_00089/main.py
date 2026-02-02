import os
import sys
import pandas as pd
import numpy as np
import torch

# Import library modules
from library.config import Config
from library.train import run_training
from library.dataset import get_data_loaders
from library.utils import compute_metric, load_checkpoint, seed_everything


def main():
    # 1. Configuration
    config = Config()
    # Override for optimized execution
    # Increasing epochs to ensure convergence (Cite solution_lesson_node_00071)
    config.EPOCHS = 80
    config.DEBUG = False

    # Ensure reproducibility
    seed_everything(config.SEED)

    print("Initializing Training Pipeline...")

    # 2. Run Training
    # This handles data loading (and caching), model init, and the training loop.
    # It saves checkpoints to config.WORKING_DIR.
    model = run_training(config)

    # 3. Load Best Model
    # run_training returns the model state at the last epoch.
    # We want the best model based on validation MAE.
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model checkpoint from {best_model_path}...")
        # Use library utility to load checkpoint
        load_checkpoint(best_model_path, model, device=config.DEVICE)
    else:
        print("\nWarning: Best model checkpoint not found. Using last epoch model.")

    # Move model to device and set to eval
    device = torch.device(config.DEVICE)
    model.to(device)
    model.eval()

    # 4. Validation & Failure Analysis
    print("\nRunning Validation Assessment...")

    # Reload loaders (fast due to caching) to get access to validation data
    _, val_loader, test_loader = get_data_loaders(config)

    val_preds = []
    val_targets = []
    val_u_out = []
    val_inputs = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            # Inference
            preds = model(x)

            # Collect data (move to CPU to save GPU memory)
            val_preds.append(preds.cpu())
            val_targets.append(y.cpu())
            val_u_out.append(u_out.cpu())
            val_inputs.append(x.cpu())

    # Flatten all tensors
    val_preds_flat = torch.cat(val_preds).view(-1)
    val_targets_flat = torch.cat(val_targets).view(-1)
    val_u_out_flat = torch.cat(val_u_out).view(-1)
    val_inputs_flat = torch.cat(val_inputs).view(-1, config.INPUT_DIM)

    # Compute Metric
    val_mae = compute_metric(val_preds_flat, val_targets_flat, val_u_out_flat)
    print(f"Final Validation Metric: {val_mae}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Filter for inspiratory phase (u_out == 0) where the metric is calculated
    mask = val_u_out_flat == 0

    if mask.sum() > 0:
        # Calculate absolute error
        errors = torch.abs(val_preds_flat[mask] - val_targets_flat[mask]).numpy()
        inputs_masked = val_inputs_flat[mask].numpy()

        print("Correlation between Error Magnitude and Input Features:")
        for i, feat_name in enumerate(config.FEATURE_COLS):
            feat_values = inputs_masked[:, i]
            # Check for constant features to avoid warnings/NaNs
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(errors, feat_values)[0, 1]
                print(f"{feat_name}: {corr:.4f}")
            else:
                print(f"{feat_name}: 0.0000 (Constant)")
    else:
        print("No inspiratory phase samples found in validation set.")

    # 5. Submission
    THRESHOLD = 0.16391726930343686

    if val_mae < THRESHOLD:
        print(
            f"\nValidation MAE ({val_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                preds = model(x)
                test_preds.append(preds.cpu())

        # Flatten predictions
        test_preds_flat = torch.cat(test_preds).view(-1).numpy()

        # Load Test Metadata to reconstruct IDs
        # The dataset loader sorts by [breath_id, id]. We must match this order.
        print("Processing submission dataframe...")
        test_df = pd.read_csv(config.TEST_PATH)

        # Sort to match the DataLoader order
        test_df_sorted = test_df.sort_values(["breath_id", "id"]).reset_index(drop=True)

        # Assign predictions
        test_df_sorted["pressure"] = test_preds_flat

        # Create submission file: id, pressure
        submission_df = test_df_sorted[["id", "pressure"]].copy()

        # Sort by ID as per sample_submission convention
        submission_df = submission_df.sort_values("id")

        # Save
        sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nValidation MAE ({val_mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
