import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.dataset import get_dataloaders
from library.model import VentilatorModel
from library.engine import Engine, evaluate


def main():
    # --- 1. Setup & Configuration ---
    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Detect device (GPU/CPU)
    device = get_device()

    # Override Config for Fast Baseline Execution
    # Reducing epochs to 15 to ensure the script completes within the time limit
    # while providing enough training to potentially meet the metric threshold.
    Config.EPOCHS = 15

    # Ensure the required submission directory exists
    os.makedirs("./submission", exist_ok=True)

    print(f"Starting execution for Experiment: {Config.EXP_ID}")
    print(f"Device: {device}")
    print(f"Training for {Config.EPOCHS} epochs")

    # --- 2. Data Loading ---
    # Load data with caching enabled to optimize runtime
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --- 3. Model Initialization ---
    print("Initializing Model...")
    model = VentilatorModel()

    # --- 4. Training ---
    # Initialize Engine and start training
    engine = Engine(model, device)
    engine.fit(train_loader, val_loader)

    # --- 5. Validation Assessment ---
    # The fit method loads the best model state at the end.
    # We perform a final evaluation to get the exact metric and data for analysis.
    print("Performing final validation assessment...")
    val_loss, val_mae = evaluate(engine.model, val_loader, device, engine.criterion)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_mae}")

    # --- 6. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    engine.model.eval()

    all_errors = []
    all_features = []

    # Collect errors and features for the inspiratory phase
    with torch.no_grad():
        for batch in val_loader:
            x, y, u_out = batch
            x = x.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            # Predict
            preds = engine.model(x)
            if preds.dim() == 3:
                preds = preds.squeeze(-1)

            # Calculate absolute error
            abs_err = torch.abs(preds - y)

            # Mask for inspiratory phase (u_out == 0)
            # Using < 0.5 for float safety
            mask = u_out < 0.5

            if mask.sum() > 0:
                # Filter data
                valid_errs = abs_err[mask]
                valid_feats = x[mask]  # Shape: (N_valid, N_features)

                all_errors.append(valid_errs.cpu().numpy())
                all_features.append(valid_feats.cpu().numpy())

    if all_errors:
        all_errors = np.concatenate(all_errors)
        all_features = np.concatenate(all_features, axis=0)

        # Calculate Pearson correlation between Error Magnitude and each Feature
        feature_names = Config.CONT_FEATURES
        correlations = []

        for i, name in enumerate(feature_names):
            feat_vals = all_features[:, i]
            # Check for constant features to avoid division by zero in correlation
            if np.std(feat_vals) > 1e-9:
                corr, _ = pearsonr(all_errors, feat_vals)
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(
            "Correlation between Error Magnitude and Input Features (Validation Set):"
        )
        for name, corr in correlations:
            print(f"  {name}: {corr:.4f}")
    else:
        print("No inspiratory phase data found for failure analysis.")

    # --- 7. Submission ---
    THRESHOLD = 0.3096454441547394

    if val_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({val_mae}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        ids, preds = engine.predict(test_loader)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": ids, "pressure": preds})

        # Sort by ID
        submission_df = submission_df.sort_values("id")

        # Save to specified path
        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nValidation Metric ({val_mae}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
