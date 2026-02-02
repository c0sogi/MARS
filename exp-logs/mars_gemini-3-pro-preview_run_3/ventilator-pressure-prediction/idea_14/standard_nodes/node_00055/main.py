import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.data_utils import engineer_features, VentilatorDataset
from library.train_utils import run_training
from library.model import MSDHNet


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    # Using full dataset and default configuration (80 epochs) as per lessons.
    # Cite solution_lesson_node_00054: Avoid aggressive data subsampling.

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Configuration: Epochs={Config.EPOCHS}, Working Dir={Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    print("Starting training pipeline...")
    # run_training will handle full dataset loading and caching automatically via get_data_loaders
    best_val_loss = run_training(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print(f"Final Validation Metric: {best_val_loss:.16f}")

    THRESHOLD = 0.1642141044139862

    if best_val_loss < THRESHOLD:
        print("\nThreshold met. Performing Failure Analysis...")

        # Load Validation Data
        val_x = np.load(os.path.join(Config.WORKING_DIR, "val_x.npy"))
        val_y = np.load(os.path.join(Config.WORKING_DIR, "val_y.npy"))

        # Load Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MSDHNet().to(device)
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        model.eval()

        # Create DataLoader for inference
        val_dataset = VentilatorDataset(val_x, val_y)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=512,  # Larger batch for faster inference
            shuffle=False,
            num_workers=2,
        )

        all_errors = []
        all_feats = []

        print("Running inference on validation set for analysis...")
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward
                preds = model(x_batch)

                # Get u_out for masking (index 1)
                u_out = x_batch[:, :, 1]

                # Calculate Absolute Error
                abs_err = torch.abs(preds - y_batch)

                # Mask: Only analyze inspiratory phase
                mask = u_out == 0

                # Flatten and filter
                # View as (-1) flattens the batch and sequence dims
                mask_flat = mask.view(-1)
                err_flat = abs_err.view(-1)
                # Flatten features: (B, L, D) -> (B*L, D)
                x_flat = x_batch.view(-1, x_batch.shape[-1])

                # Select valid points
                valid_err = err_flat[mask_flat].cpu().numpy()
                valid_feats = x_flat[mask_flat].cpu().numpy()

                all_errors.append(valid_err)
                all_feats.append(valid_feats)

        # Concatenate
        all_errors = np.concatenate(all_errors)
        all_feats = np.concatenate(all_feats)

        print(f"Analyzed {len(all_errors)} time steps.")
        print("\nCorrelation between Error Magnitude and Input Features:")
        print("-" * 50)
        print(f"{'Feature':<20} | {'Correlation':<10}")
        print("-" * 50)

        for i, feat_name in enumerate(Config.FEATURE_COLS):
            # Calculate Pearson correlation
            corr = np.corrcoef(all_feats[:, i], all_errors)[0, 1]
            print(f"{feat_name:<20} | {corr:.6f}")
        print("-" * 50)

        print(f"Submission file generated at: {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {best_val_loss} did not meet threshold {THRESHOLD}."
        )
        print("Discarding submission file.")
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
