import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_fold_dataloaders, get_test_dataloader
from library.train_eval import train_fold, predict
from library.model import WBMGNet


def run():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    print("Initializing WBMG-Net Pipeline...")
    seed_everything(Config.SEED)
    Config.create_directories()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override Config for Fast Baseline execution
    # Reducing epochs to ensure completion within time limit while maintaining performance
    Config.NUM_EPOCHS = 30
    Config.PATIENCE = 8

    # ==========================================
    # 2. 5-Fold Cross-Validation Training
    # ==========================================
    # Lists to store Out-Of-Fold (OOF) data for global evaluation
    all_val_preds = []
    all_val_targets = []
    all_val_angles = []

    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold_idx in range(Config.NUM_FOLDS):
        print(f"\n--- Processing Fold {fold_idx} ---")

        # Get DataLoaders with strict fold-wise preprocessing
        # stats contains min_vals, max_vals, angle_mean for this specific fold
        train_loader, val_loader, stats = get_fold_dataloaders(
            fold_idx, load_cached_data=True
        )

        # Train the model
        # train_fold handles the loop, early stopping, and returns the model with best weights loaded
        model, history = train_fold(fold_idx, train_loader, val_loader, device)

        # Generate OOF Predictions on Validation Set
        # predict() returns flattened numpy array of probabilities
        val_probs = predict(model, val_loader, device)

        # Extract targets and angles from val_loader to align with predictions
        # val_loader has shuffle=False, so order is preserved
        fold_targets = []
        fold_angles = []
        for _, angles, labels in val_loader:
            fold_targets.extend(labels.numpy())
            fold_angles.extend(angles.numpy())

        # Accumulate results
        all_val_preds.extend(val_probs)
        all_val_targets.extend(fold_targets)
        all_val_angles.extend(fold_angles)

        # Clean up to save memory
        del model, train_loader, val_loader
        torch.cuda.empty_cache()

    # ==========================================
    # 3. Global Validation & Failure Analysis
    # ==========================================
    print("\n" + "=" * 30)
    print("EVALUATION REPORT")
    print("=" * 30)

    y_true = np.array(all_val_targets)
    y_pred = np.array(all_val_preds)
    val_angles = np.array(all_val_angles)

    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Calculate Log Loss
    final_metric = log_loss(y_true, y_pred_clipped)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n[Failure Analysis]")
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Correlation between Error and Incidence Angle
    # Check variance to avoid division by zero
    if np.std(val_angles) > 0:
        corr_angle = np.corrcoef(errors, val_angles)[0, 1]
        print(f"Correlation between Error and Incidence Angle: {corr_angle:.10f}")
    else:
        print("Incidence angle variance is 0, cannot compute correlation.")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    threshold = 0.16676861786296204

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} meets threshold ({threshold}). Generating submission..."
        )

        test_preds_sum = None
        test_ids_list = None

        for fold_idx in range(Config.NUM_FOLDS):
            print(f"Inference on Test Set using Fold {fold_idx} model...")

            # 1. Retrieve normalization stats for this fold
            # We call get_fold_dataloaders again just to get the 'stats' dict
            # This ensures we normalize test data exactly how this fold's training data was normalized
            _, _, stats = get_fold_dataloaders(fold_idx, load_cached_data=True)

            # 2. Create Test Loader
            test_loader, ids = get_test_dataloader(stats, load_cached_data=True)

            if test_ids_list is None:
                test_ids_list = ids

            # 3. Load Model
            model = CDPNet().to(device)
            checkpoint_path = Config.get_checkpoint_path(fold_idx)

            # Load checkpoint
            if not os.path.exists(checkpoint_path):
                print(
                    f"Warning: Checkpoint {checkpoint_path} not found. Skipping fold."
                )
                continue

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            # 4. Predict
            preds = predict(model, test_loader, device)

            # 5. Accumulate
            if test_preds_sum is None:
                test_preds_sum = preds
            else:
                test_preds_sum += preds

            # Clean up
            del model, test_loader
            torch.cuda.empty_cache()

        # Average predictions
        avg_preds = test_preds_sum / Config.NUM_FOLDS

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids_list, "is_iceberg": avg_preds})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
