import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data_loader import get_data_loaders, process_data
from library.model import CSPHN
from library.trainer import run_fold


def main():
    # 1. Configuration & Setup
    # Use config epochs for full convergence (Cite solution_lesson_node_00023)
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(
        f"Execution Config: Device={device}, Epochs={Config.NUM_EPOCHS}, Folds={Config.N_FOLDS}"
    )

    # Initialize storage for OOF predictions
    oof_preds = []
    oof_targets = []
    oof_angles = []
    oof_image_means = []

    model_paths = []

    # 2. Training Loop (5-Fold CV with OOF Evaluation)
    print("\n=== Starting 5-Fold Cross-Validation (With OOF Evaluation) ===")

    for fold in range(Config.N_FOLDS):
        # A. Train the fold
        run_fold(fold_idx=fold, load_cached_data=True)

        # B. Load the best model for this fold
        checkpoint_path = os.path.join(Config.WORK_DIR, f"csphn_model_fold_{fold}.pth")
        model_paths.append(checkpoint_path)

        model = CSPHN().to(device)
        load_checkpoint(checkpoint_path, model)
        model.eval()

        # C. Get Validation Data for this fold ONLY
        # This ensures we validate on data NOT seen during training of this fold.
        _, val_loader, _ = get_data_loaders(fold=fold, load_cached_data=True)

        print(f"Generating OOF predictions for Fold {fold}...")

        # D. Predict on Validation Set
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles_dev = angles.to(device)

                # Single model prediction
                out = model(images, angles_dev)

                # Store predictions and metadata
                oof_preds.extend(out.cpu().numpy().flatten())
                oof_targets.extend(labels.numpy().flatten())
                oof_angles.extend(angles.numpy())

                # For failure analysis
                img_means = images.cpu().numpy().mean(axis=(1, 2, 3))
                oof_image_means.extend(img_means)

    # Convert collected lists to numpy arrays
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)
    oof_angles = np.array(oof_angles)
    oof_image_means = np.array(oof_image_means)

    # 3. Calculate OOF Metric
    # Clip predictions to avoid log(0)
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(oof_targets, oof_preds_clipped)

    print(f"\nFinal OOF Validation Metric: {final_metric}")

    # 4. Failure Analysis on OOF Data
    print("\n=== Failure Analysis (OOF) ===")
    errors = np.abs(oof_targets - oof_preds)

    # Correlation with Incidence Angle
    valid_angle_mask = ~np.isnan(oof_angles)
    if np.sum(valid_angle_mask) > 1:
        corr_angle, _ = pearsonr(errors[valid_angle_mask], oof_angles[valid_angle_mask])
        print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    else:
        print("Correlation (Error vs Inc Angle): N/A (Insufficient data)")

    # Correlation with Signal Intensity
    corr_intensity, _ = pearsonr(errors, oof_image_means)
    print(f"Correlation (Error vs Signal Intensity): {corr_intensity:.4f}")

    # 5. Submission Generation
    threshold = 0.17493283735739185
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        # Load Test Data
        _, _, test_loader = get_data_loaders(fold=None, load_cached_data=True)

        # Get Test IDs
        data_dict = process_data(load_cached_data=True)
        test_ids = data_dict["ids_test"]

        # Reload all models for Ensemble Prediction
        models = []
        for path in model_paths:
            model = CSPHN().to(device)
            load_checkpoint(path, model)
            model.eval()
            models.append(model)

        test_preds = []

        with torch.no_grad():
            for images, angles, _ in test_loader:
                images = images.to(device)
                angles_dev = angles.to(device)

                batch_preds = []
                for model in models:
                    out = model(images, angles_dev)
                    batch_preds.append(out.cpu().numpy())

                # Average predictions across models (Bagging)
                avg_preds = np.mean(batch_preds, axis=0)
                test_preds.extend(avg_preds)

        test_preds = np.array(test_preds).flatten()

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission.head())

    else:
        print(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
