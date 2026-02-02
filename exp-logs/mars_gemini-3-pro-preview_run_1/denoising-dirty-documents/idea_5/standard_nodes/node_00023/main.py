import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import SEED, WORKING_DIR, VAL_METADATA, DEVICE, set_seed, EPOCHS
from library.train import run_training
from library.inference import generate_submission, predict_tta
from library.model import ResidualShallowUNet
from library.dataset import _load_and_cache_data, DenoisingDataset
from library.utils import worker_init_fn


def main():
    # Ensure reproducibility
    set_seed(SEED)

    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    # We run full training to meet the performance threshold.
    # We run all 5 folds to ensure the ensemble logic in inference works correctly.
    print(f"Starting full training (5 folds, {EPOCHS} epochs each)...")
    run_training(epochs=EPOCHS, n_folds=5, load_cached_data=True)

    # -------------------------------------------------------------------------
    # 2. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("Loading validation data for analysis...")
    # Load specifically the validation set defined in metadata
    val_data = _load_and_cache_data(
        [VAL_METADATA], "val_analysis_cache.npz", load_cached_data=True
    )

    # Create Dataset and Loader (mode='val' returns full images, no cropping)
    val_ds = DenoisingDataset(val_data, mode="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    # Load the trained ensemble models
    models = []
    for fold_idx in range(5):
        model_path = os.path.join(WORKING_DIR, f"model_fold_{fold_idx}.pth")
        if os.path.exists(model_path):
            model = ResidualShallowUNet(n_channels=1, n_classes=1)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models.append(model)

    if not models:
        print("Error: No models were trained.")
        return

    print(f"Loaded {len(models)} models for validation.")

    # Run Inference
    total_sse = 0.0
    total_pixels = 0

    image_rmses = []
    feat_means = []
    feat_stds = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, (noisy, residual_target) in enumerate(val_loader):
            # noisy: (1, 1, H, W)
            # residual_target: (1, 1, H, W)

            # 1. Extract Features for Failure Analysis
            # Convert to numpy for stats
            noisy_np = noisy.squeeze().numpy()
            feat_means.append(np.mean(noisy_np))
            feat_stds.append(np.std(noisy_np))

            # 2. Ensemble Prediction
            fold_residuals = []
            for model in models:
                # predict_tta handles device transfer internally for input
                # returns cpu tensor
                res = predict_tta(model, noisy, DEVICE)
                fold_residuals.append(res)

            # Average predictions
            avg_residual = torch.stack(fold_residuals).mean(dim=0)  # (1, 1, H, W)

            # 3. Compute Metrics
            # Reconstruct Clean: Clean = Noisy - Predicted Residual
            # Note: noisy is from loader (CPU), avg_residual is CPU
            pred_clean = noisy - avg_residual

            # Target Clean: Clean = Noisy - True Residual
            target_clean = noisy - residual_target

            # Clamp to valid range
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)
            target_clean = torch.clamp(target_clean, 0.0, 1.0)

            # Squared Error
            diff = pred_clean - target_clean
            sse = torch.sum(diff**2).item()
            numel = torch.numel(diff)

            total_sse += sse
            total_pixels += numel

            # Per-image RMSE
            image_rmses.append(np.sqrt(sse / numel))

    # Compute Global RMSE
    final_metric = np.sqrt(total_sse / total_pixels)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    # Use numpy for correlation to avoid scipy dependency issues if any
    if len(image_rmses) > 1:
        corr_mean = np.corrcoef(image_rmses, feat_means)[0, 1]
        corr_std = np.corrcoef(image_rmses, feat_stds)[0, 1]
    else:
        corr_mean = 0.0
        corr_std = 0.0

    print("-" * 30)
    print("Failure Analysis:")
    print(f"Correlation (Error vs Input Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Input Std): {corr_std:.4f}")
    print("-" * 30)

    # -------------------------------------------------------------------------
    # 3. Submission Logic
    # -------------------------------------------------------------------------
    THRESHOLD = 0.012221260240721992

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
