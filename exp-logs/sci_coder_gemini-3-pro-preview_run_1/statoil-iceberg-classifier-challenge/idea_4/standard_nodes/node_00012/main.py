import os
import sys
import numpy as np
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_fold_loaders
from library.model import IcebergResNet34
from library.trainer import run_fold
from library.inference import generate_ensemble_predictions


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 20

    print(f"Training with {Config.NUM_FOLDS} folds, {Config.NUM_EPOCHS} epochs each.")

    # 2. Training and OOF Generation
    model_paths = []
    oof_preds = []
    oof_targets = []
    oof_angles = []

    for fold_idx in range(Config.NUM_FOLDS):
        # Train the model for the current fold
        # run_fold saves the best model and returns its path
        model_path = run_fold(fold_idx, load_cached_data=True)
        model_paths.append(model_path)

        # Generate OOF predictions for validation
        print(f"[Fold {fold_idx}] Generating OOF predictions...")

        # Get validation loader for this fold
        _, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

        # Load the best model for this fold
        model = IcebergResNet34()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_preds = []
        fold_targets = []

        # Inference on validation set
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                logits = model(images, angles)
                probs = torch.sigmoid(logits)

                fold_preds.extend(probs.cpu().numpy().flatten())
                fold_targets.extend(labels.cpu().numpy().flatten())

        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)

        # Collect incidence angles for failure analysis
        # val_loader.dataset.angles contains the angles in order (shuffle=False)
        oof_angles.extend(val_loader.dataset.angles)

    # 3. Global Validation Metric
    y_true = np.array(oof_targets)
    y_pred = np.array(oof_preds)

    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred_clipped)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(y_true - y_pred)

    if len(oof_angles) > 0:
        corr, _ = pearsonr(errors, oof_angles)
        print(f"Correlation between Error and Incidence Angle: {corr:.6f}")
    else:
        print("Incidence angle data not available for analysis.")

    # 5. Submission
    THRESHOLD = 0.21099163245555455
    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_ensemble_predictions(model_paths)
    else:
        print(
            f"Metric {final_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
