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
    # Override epochs for fast baseline execution as per requirements
    Config.NUM_EPOCHS = 20
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(
        f"Execution Config: Device={device}, Epochs={Config.NUM_EPOCHS}, Folds={Config.N_FOLDS}"
    )

    # 2. Training Loop (5-Fold CV)
    print("\n=== Starting 5-Fold Cross-Validation Training ===")
    model_paths = []
    for fold in range(Config.N_FOLDS):
        val_loss = run_fold(fold_idx=fold, load_cached_data=True)
        checkpoint_path = os.path.join(Config.WORK_DIR, f"csphn_model_fold_{fold}.pth")
        model_paths.append(checkpoint_path)

    # 3. Ensemble Validation on Hold-Out Set
    print("\n=== Running Ensemble Validation on Fixed Hold-Out Set ===")
    # Load fixed validation split (fold=None returns fixed split loaders)
    _, val_loader, _ = get_data_loaders(fold=None, load_cached_data=True)

    # Initialize models
    models = []
    for path in model_paths:
        model = CSPHN().to(device)
        load_checkpoint(path, model)
        model.eval()
        models.append(model)

    val_preds = []
    val_targets = []
    val_angles = []
    val_image_means = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles_dev = angles.to(device)

            # Ensemble Prediction
            batch_preds = []
            for model in models:
                # Ensure model expects (images, angles)
                out = model(images, angles_dev)
                batch_preds.append(out.cpu().numpy())

            # Average predictions across models
            avg_preds = np.mean(batch_preds, axis=0)

            val_preds.extend(avg_preds)
            val_targets.extend(labels.numpy())
            val_angles.extend(angles.numpy())

            # Calculate image mean intensity for failure analysis
            # images shape: (B, 3, 75, 75)
            img_means = images.cpu().numpy().mean(axis=(1, 2, 3))
            val_image_means.extend(img_means)

    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()
    val_angles = np.array(val_angles)
    val_image_means = np.array(val_image_means)

    # Calculate Metric
    # Clip predictions to avoid log(0)
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(val_targets, val_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(val_targets - val_preds)

    # Correlation with Incidence Angle
    # Handle potential NaNs in angles if any (though loader imputes them)
    valid_angle_mask = ~np.isnan(val_angles)
    if np.sum(valid_angle_mask) > 1:
        corr_angle, _ = pearsonr(errors[valid_angle_mask], val_angles[valid_angle_mask])
        print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    else:
        print("Correlation (Error vs Inc Angle): N/A (Insufficient data)")

    # Correlation with Signal Intensity
    corr_intensity, _ = pearsonr(errors, val_image_means)
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

        test_preds = []

        with torch.no_grad():
            for images, angles, _ in test_loader:
                images = images.to(device)
                angles_dev = angles.to(device)

                batch_preds = []
                for model in models:
                    out = model(images, angles_dev)
                    batch_preds.append(out.cpu().numpy())

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
