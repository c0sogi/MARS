import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, cleanup_cache, get_device
from library.dataset import prepare_data
from library.model import FMDHNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline Execution (Target < 2 hours)
    # We use a subset of data and fewer epochs to demonstrate the pipeline quickly.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 15000  # ~1.2M samples, sufficient for convergence demo
    Config.EPOCHS = 6  # Reduced from 80 to 6 for speed
    Config.BATCH_SIZE = 128  # Keep batch size small as per "Critical Mass"

    # Reproducibility
    seed_everything(Config.SEED)

    # Clean cache to ensure we process the debug subset, not load old full data
    cleanup_cache()

    print(
        f"Running Fast Baseline with DEBUG={Config.DEBUG}, "
        f"Samples={Config.DEBUG_SAMPLE_SIZE}, Epochs={Config.EPOCHS}"
    )

    # =========================================================================
    # 2. Data Preparation
    # =========================================================================
    # load_cached_data=False forces regeneration of data with the DEBUG settings
    train_loader, val_loader, test_loader, test_ids = prepare_data(
        load_cached_data=False
    )

    # =========================================================================
    # 3. Model Initialization & Training
    # =========================================================================
    print("Initializing FMDH-Net...")
    model = FMDHNet()

    print("Starting Training...")
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader)

    # =========================================================================
    # 4. Validation Assessment
    # =========================================================================
    print("Performing Final Validation...")
    val_mae = trainer.validate(val_loader)

    # STRICT OUTPUT FORMAT REQUIRED
    print(f"Final Validation Metric: {val_mae}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("Running Failure Analysis...")
    model.eval()
    device = get_device()

    all_errors = []
    all_features = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            # Extract u_out (index 2) to mask expiratory phase
            u_out = x[:, :, 2]
            mask = u_out == 0

            # Forward pass
            preds = model(x).squeeze(-1)

            # Calculate Absolute Error
            abs_error = torch.abs(preds - y)

            # Apply Mask
            masked_error = abs_error[mask]
            masked_features = x[mask]

            if masked_error.numel() > 0:
                all_errors.append(masked_error.cpu().numpy())
                all_features.append(masked_features.cpu().numpy())

    if all_errors:
        # Concatenate all batches
        flat_errors = np.concatenate(all_errors)
        flat_features = np.concatenate(all_features)

        # Create DataFrame for correlation calculation
        feat_df = pd.DataFrame(flat_features, columns=Config.FEATURE_COLS)
        feat_df["error_magnitude"] = flat_errors

        # Calculate correlation with error magnitude
        correlations = (
            feat_df.corr()["error_magnitude"]
            .drop("error_magnitude")
            .sort_values(ascending=False)
        )

        print(
            "\n=== Failure Analysis: Correlation of Features with Error Magnitude ==="
        )
        print(correlations)
        print("====================================================================")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    THRESHOLD = 0.1642141044139862

    if val_mae < THRESHOLD:
        print(f"\nValidation Metric {val_mae} is below threshold {THRESHOLD}.")
        print("Generating submission file...")
        trainer.predict(test_loader, test_ids)
    else:
        print(f"\nValidation Metric {val_mae} is NOT below threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
