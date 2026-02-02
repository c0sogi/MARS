import pandas as pd
import numpy as np
import torch
import os
import sys
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data_factory import prepare_datasets
from library.trainer import Trainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for fast baseline execution within time limits
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 128

    # Ensure working directories exist
    Config.setup_dirs()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("Preparing datasets...")
    # Load cached data if available to save time
    train_dataset, val_dataset, test_dataset, scaler = prepare_datasets(
        load_cached_data=True
    )

    # Create DataLoaders
    # num_workers set to 4 to utilize available vCPUs efficiently
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training...")
    trainer.fit(train_loader, val_loader)

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    print("\nPerforming Validation and Failure Analysis...")

    # Load the best model saved during training
    if os.path.exists(trainer.best_model_path):
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    trainer.model.eval()

    val_preds = []
    val_targets = []
    val_u_out = []
    val_inputs = []

    # Optimized inference loop
    with torch.no_grad():
        for X, y, u_out in val_loader:
            X = X.to(trainer.device)
            # No need to move y/u_out to GPU for metric calc, but needed if loss calc was here.
            # We just need them for CPU-based metric/analysis.

            preds = trainer.model(X)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.numpy())
            val_u_out.append(u_out.numpy())
            val_inputs.append(X.cpu().numpy())

    # Flatten results
    # Shape: (N_batches, B, Seq, 1) -> (Total_Samples,)
    y_pred_flat = np.concatenate(val_preds, axis=0).squeeze(-1).flatten()
    y_true_flat = np.concatenate(val_targets, axis=0).flatten()
    u_out_flat = np.concatenate(val_u_out, axis=0).flatten()

    # Shape: (N_batches, B, Seq, Feats) -> (Total_Samples, Feats)
    X_flat = np.concatenate(val_inputs, axis=0)
    X_flat = X_flat.reshape(-1, X_flat.shape[-1])

    # Compute Metric
    final_metric = compute_metric(y_pred_flat, y_true_flat, u_out_flat)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    abs_errors = np.abs(y_pred_flat - y_true_flat)

    # Mask for inspiratory phase (u_out == 0)
    insp_mask = u_out_flat == 0

    if np.sum(insp_mask) > 0:
        errors_insp = abs_errors[insp_mask]
        features_insp = X_flat[insp_mask]

        print("Correlation between Error Magnitude and Features (Inspiratory Phase):")
        correlations = []
        for i, feat_name in enumerate(Config.FEATURE_COLS):
            feat_values = features_insp[:, i]

            # Avoid correlation calculation if feature is constant
            if np.std(feat_values) < 1e-9:
                corr = 0.0
            else:
                corr = np.corrcoef(errors_insp, feat_values)[0, 1]

            correlations.append((feat_name, corr))
            print(f"{feat_name}: {corr:.4f}")

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print("\nTop 3 Features most correlated with Error:")
        for name, corr in correlations[:3]:
            print(f"{name}: {corr:.4f}")
    else:
        print("No inspiratory phase data found for analysis.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.1642141044139862

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        predictions = trainer.predict(test_loader)

        # Load sample submission
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Verify lengths
        if len(predictions) != len(sample_sub):
            print(
                f"Warning: Prediction length ({len(predictions)}) does not match sample submission ({len(sample_sub)})."
            )

        # Assign predictions
        sample_sub["pressure"] = predictions

        # Save submission
        sample_sub.to_csv(Config.OUTPUT_PATH, index=False)
        print(f"Submission saved to {Config.OUTPUT_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
