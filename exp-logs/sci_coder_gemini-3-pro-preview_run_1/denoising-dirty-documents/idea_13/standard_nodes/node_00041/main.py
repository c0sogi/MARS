import os
import time
import numpy as np
import torch
import pandas as pd

from library.config import Config
from library.utils import seed_everything, rmse_score
from library.dataset import get_train_val_datasets
from library.train import train_one_seed
from library.model import UNet
from library.inference import predict_and_submit, apply_tta


def main():
    # 1. Setup and Configuration
    seed_everything(42)

    # Fast Baseline Configuration
    # We reduce epochs to 200 to ensure rapid execution while maintaining
    # sufficient convergence for the small dataset (92 images).
    FAST_EPOCHS = 200

    print(f"Initializing Fast Baseline Run with {FAST_EPOCHS} epochs per seed...")

    # 2. Data Loading
    # Load datasets with caching enabled for speed
    train_ds, val_ds = get_train_val_datasets(load_cached_data=True)

    # 3. Ensemble Training
    print(f"Starting training for ensemble of {len(Config.SEEDS)} models...")
    start_time = time.time()

    for seed in Config.SEEDS:
        train_one_seed(
            seed=seed,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=FAST_EPOCHS,
            batch_size=Config.BATCH_SIZE,
            device=Config.DEVICE,
            num_workers=Config.NUM_WORKERS,
        )

    print(f"Ensemble training completed in {time.time() - start_time:.2f} seconds.")

    # 4. Validation and Failure Analysis
    print("Starting Validation and Failure Analysis...")
    device = Config.DEVICE

    # Load all trained models into memory
    models = []
    for seed in Config.SEEDS:
        model_path = Config.get_model_path(seed)
        if os.path.exists(model_path):
            model = UNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model for seed {seed} not found.")

    if not models:
        print("No models available for validation. Exiting.")
        return

    # Prepare for validation inference
    # We iterate sequentially to apply TTA and Ensemble averaging
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    all_preds_flat = []
    all_targets_flat = []

    # Metrics for failure analysis
    img_errors = []
    feat_means = []
    feat_stds = []

    with torch.no_grad():
        for i, (noisy_padded, clean_padded, img_id) in enumerate(val_loader):
            noisy_padded = noisy_padded.to(device)

            # Retrieve original unpadded data for accurate metric calculation
            # val_ds.clean_imgs contains the inverted, normalized original images
            orig_clean_np = val_ds.clean_imgs[i]
            orig_noisy_np = val_ds.noisy_imgs[i]
            h_orig, w_orig = orig_clean_np.shape

            # Collect Input Features (on original noisy image)
            feat_means.append(np.mean(orig_noisy_np))
            feat_stds.append(np.std(orig_noisy_np))

            # Ensemble Inference with TTA
            model_preds = []
            for model in models:
                # apply_tta returns (1, 1, H_pad, W_pad)
                pred_tensor = apply_tta(model, noisy_padded, device)
                model_preds.append(pred_tensor)

            # Average predictions across the ensemble
            avg_pred_tensor = torch.stack(model_preds).mean(dim=0)

            # Move to CPU and convert to numpy
            pred_padded_np = avg_pred_tensor.squeeze().cpu().numpy()

            # Unpad prediction to match original dimensions
            # Albumentations PadIfNeeded centers the image
            h_pad, w_pad = pred_padded_np.shape
            diff_h = h_pad - h_orig
            diff_w = w_pad - w_orig
            top = diff_h // 2
            left = diff_w // 2

            pred_np = pred_padded_np[top : top + h_orig, left : left + w_orig]

            # Accumulate flat arrays for global RMSE
            all_preds_flat.append(pred_np.flatten())
            all_targets_flat.append(orig_clean_np.flatten())

            # Calculate per-image RMSE for failure analysis
            img_rmse = np.sqrt(np.mean((orig_clean_np - pred_np) ** 2))
            img_errors.append(img_rmse)

    # Calculate Final Validation Metric
    y_pred_global = np.concatenate(all_preds_flat)
    y_true_global = np.concatenate(all_targets_flat)

    final_rmse = np.sqrt(np.mean((y_true_global - y_pred_global) ** 2))

    # Print Metric (Required Format)
    print(f"Final Validation Metric: {final_rmse}")

    # Calculate Correlations for Failure Analysis
    if len(img_errors) > 1:
        corr_mean = np.corrcoef(img_errors, feat_means)[0, 1]
        corr_std = np.corrcoef(img_errors, feat_stds)[0, 1]

        print("-" * 30)
        print("Failure Analysis Report")
        print("-" * 30)
        print(f"Correlation (Error vs Input Mean): {corr_mean}")
        print(f"Correlation (Error vs Input Std): {corr_std}")
        print("-" * 30)

    # 5. Submission Generation
    THRESHOLD = 0.011870221132053216

    if final_rmse < THRESHOLD:
        print(f"Validation metric {final_rmse} meets threshold {THRESHOLD}.")
        print("Generating submission file...")
        predict_and_submit(load_cached_data=True)
    else:
        print(f"Validation metric {final_rmse} does not meet threshold {THRESHOLD}.")
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
