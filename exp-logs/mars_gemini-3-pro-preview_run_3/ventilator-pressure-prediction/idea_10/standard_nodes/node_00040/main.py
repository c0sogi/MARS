import sys
import os
import numpy as np
import pandas as pd
import torch

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device, compute_mae
from library.data_loader import get_data_loaders
from library.trainer import Trainer
from library.model import NCPNet


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline Execution
    # Limit epochs to ensure completion within 2 hours on A100
    Config.EPOCHS = 10

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Execution Config: Epochs={Config.EPOCHS}, Device={device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    # Load data (handles caching, feature engineering, and scaling internally)
    # We use debug=False to use the full dataset for competitive performance,
    # relying on the A100 GPU and reduced epochs for speed.
    train_loader, val_loader, test_loader = get_data_loaders(debug=False)

    # =========================================================================
    # 3. Model Training
    # =========================================================================
    trainer = Trainer()
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # =========================================================================
    # 4. Validation & Failure Analysis
    # =========================================================================
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model checkpoint
    trainer.load_best_model()
    model = trainer.model
    model.eval()

    # Containers for analysis
    all_errors = []
    all_features = []

    # Accumulators for global metric
    total_abs_error = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Inference
            preds = model(x)

            # Flatten for processing
            preds_flat = preds.view(-1)
            y_flat = y.view(-1)
            u_out_flat = u_out.view(-1)

            # Mask: Only consider inspiratory phase (u_out == 0)
            mask = u_out_flat == 0

            if mask.sum() > 0:
                # Calculate errors
                diff = torch.abs(preds_flat - y_flat)
                valid_errors = diff[mask]

                # Update global metric accumulators
                total_abs_error += valid_errors.sum().item()
                total_count += valid_errors.numel()

                # Store for failure analysis (move to CPU numpy)
                all_errors.append(valid_errors.cpu().numpy())

                # Extract corresponding features
                # x is (Batch, Seq, Feats) -> Flatten -> Filter by mask
                x_flat = x.view(-1, x.shape[-1])
                valid_feats = x_flat[mask]
                all_features.append(valid_feats.cpu().numpy())

    # Compute Final Metric
    final_metric = total_abs_error / total_count if total_count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if len(all_errors) > 0:
        print("\n=== Failure Analysis ===")
        # Concatenate all batches
        errors_arr = np.concatenate(all_errors)
        features_arr = np.concatenate(all_features)

        # Create DataFrame
        analysis_df = pd.DataFrame(features_arr, columns=Config.FEATURE_COLS)
        analysis_df["error"] = errors_arr

        # Compute correlations
        correlations = analysis_df.corr()["error"].drop("error")
        print("Correlation between Error Magnitude and Features:")
        print(correlations.sort_values(ascending=False))

    # =========================================================================
    # 5. Submission
    # =========================================================================
    THRESHOLD = 0.22291307151317596

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions
        # trainer.predict returns a flat numpy array of all predictions
        predictions = trainer.predict(test_loader)

        # Load Test IDs from cache (saved during data loading)
        if os.path.exists(Config.CACHE_TEST_IDS):
            test_ids = np.load(Config.CACHE_TEST_IDS)
        else:
            # Fallback: Read from metadata if cache missing (unlikely)
            test_df = pd.read_csv(Config.TEST_PATH)
            test_ids = test_df[Config.ID_COL].values

        # Sanity check lengths
        if len(predictions) != len(test_ids):
            print(
                f"Warning: Length mismatch. Preds: {len(predictions)}, IDs: {len(test_ids)}"
            )
            # Truncate to match if necessary (though they should match)
            min_len = min(len(predictions), len(test_ids))
            predictions = predictions[:min_len]
            test_ids = test_ids[:min_len]

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "pressure": predictions})

        # Save
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
