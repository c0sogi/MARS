import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_fold_loaders, get_test_loader
from library.model import PPCWBN
from library.train_eval import run_fold


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Store OOF data for global evaluation
    oof_preds = []
    oof_targets = []
    oof_angles = []

    # 2. Train and Validate Folds
    for fold_idx in range(Config.NUM_FOLDS):
        print(f"\n{'='*20} Processing Fold {fold_idx} {'='*20}")

        # Train the model for this fold
        result = run_fold(fold_idx)

        # Load the best model for inference
        model = PPCWBN().to(device)
        model.load_state_dict(torch.load(result["model_path"], map_location=device))
        model.eval()

        # Get validation loader for this fold to generate OOF predictions
        # Note: We don't need the training loader here
        _, val_loader, _, _ = get_fold_loaders(fold_idx, load_cached_data=True)

        fold_probs = []
        fold_targets = []
        fold_angles = []

        # Inference on validation set
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                outputs = model(images, angles_gpu)
                probs = torch.sigmoid(outputs).view(-1).cpu().numpy()

                fold_probs.extend(probs)
                fold_targets.extend(labels.numpy())
                fold_angles.extend(angles.numpy())

        oof_preds.extend(fold_probs)
        oof_targets.extend(fold_targets)
        oof_angles.extend(fold_angles)

    # 3. Global Validation Metric
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)
    oof_angles = np.array(oof_angles)

    # Calculate Log Loss on the entire dataset (OOF)
    global_loss = log_loss(oof_targets, oof_preds, labels=[0, 1])
    print(f"\nFinal Validation Metric: {global_loss}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(oof_targets - oof_preds)

    # Correlation with Incidence Angle
    # Note: angles might have been imputed, but that's what the model saw
    corr_angle = np.corrcoef(errors, oof_angles)[0, 1]
    print(f"Correlation (Error vs Inc_Angle): {corr_angle}")

    # Error by Class
    mean_err_ship = np.mean(errors[oof_targets == 0])
    mean_err_ice = np.mean(errors[oof_targets == 1])
    print(f"Mean Error (Ship - Class 0): {mean_err_ship:.4f}")
    print(f"Mean Error (Iceberg - Class 1): {mean_err_ice:.4f}")

    # 5. Submission Generation
    THRESHOLD = 0.16676861786296204

    if global_loss < THRESHOLD:
        print(
            f"\nValidation score meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds_sum = None
        test_ids = None

        for fold_idx in range(Config.NUM_FOLDS):
            print(f"Inference on Test Set with Fold {fold_idx} model...")

            # Load scaling stats for this fold
            stats_path = os.path.join(Config.WORKING_DIR, f"stats_fold_{fold_idx}.json")
            with open(stats_path, "r") as f:
                stats_data = json.load(f)

            scaling_stats = stats_data["scaling_stats"]
            angle_mean = stats_data["angle_mean"]

            # Get Test Loader
            test_loader, ids = get_test_loader(
                scaling_stats, angle_mean, load_cached_data=True
            )
            if test_ids is None:
                test_ids = ids

            # Load Model
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
            model = PPCWBN().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_test_preds = []

            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
                    fold_test_preds.extend(probs)

            fold_test_preds = np.array(fold_test_preds)

            if test_preds_sum is None:
                test_preds_sum = fold_test_preds
            else:
                test_preds_sum += fold_test_preds

        # Average predictions
        avg_preds = test_preds_sum / Config.NUM_FOLDS

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score {global_loss} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
