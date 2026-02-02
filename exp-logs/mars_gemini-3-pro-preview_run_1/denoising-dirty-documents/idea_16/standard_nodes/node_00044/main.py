import pandas as pd
import numpy as np
import torch
import os
import sys

# Import library modules
import library.config
import library.train_engine
import library.inference_engine
import library.utils
import library.model
import library.dataset

# --- Configuration Override for Fast Baseline ---
# The default 1000 epochs is too long for the 25-minute limit with 10 models.
# We reduce it to 50 epochs, which is sufficient for a baseline on this small dataset (92 images).
# We modify the variable directly in the train_engine module where it is used.
library.train_engine.EPOCHS = 50
print(f"Configuration overridden: EPOCHS set to {library.train_engine.EPOCHS}")

# --- Imports from Library ---
from library.config import (
    STREAM_A_CONFIG,
    STREAM_B_CONFIG,
    VAL_METADATA_PATH,
    WORKING_DIR,
    DEVICE,
    SUBMISSION_FILE,
)
from library.train_engine import train_model
from library.inference_engine import (
    load_ensemble_models,
    predict_test_set,
    apply_tta,
    reverse_tta,
)
from library.dataset import DenoisingDataset
from library.utils import calculate_rmse, get_device
from torch.utils.data import DataLoader


def main():
    # 1. Train Ensemble
    # Stream A (Context Specialists)
    print("Training Stream A models...")
    for i in range(len(STREAM_A_CONFIG["seeds"])):
        train_model(STREAM_A_CONFIG, i)

    # Stream B (Diversity Specialists)
    print("Training Stream B models...")
    for i in range(len(STREAM_B_CONFIG["seeds"])):
        train_model(STREAM_B_CONFIG, i)

    # 2. Validation & Failure Analysis
    print("Starting Validation...")
    device = get_device()

    # Load Validation Data
    val_df = pd.read_csv(VAL_METADATA_PATH)
    val_dataset = DenoisingDataset(
        val_df,
        img_size=None,
        augment=False,
        cache_name="val_cache",
        load_cached_data=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load Models
    models = load_ensemble_models(device)

    # Inference on Validation
    y_true = []
    y_pred = []

    # For failure analysis
    errors = []
    means = []
    stds = []

    # TTA Settings
    # We access via library.config to ensure we get the values used by the library
    tta_enabled = library.config.TTA_ENABLED
    tta_views = library.config.TTA_VIEWS

    with torch.no_grad():
        for i, (noisy_t, clean_t, img_id) in enumerate(val_loader):
            noisy_t = noisy_t.to(device)
            # clean_t from loader is padded; we will use the original from dataset dict for metric calculation

            # Ensemble Prediction
            ensemble_accum = None
            count = 0

            for model in models:
                views = range(tta_views) if tta_enabled else [0]
                for k in views:
                    inputs = apply_tta(noisy_t, k)
                    outputs = model(inputs)
                    outputs = reverse_tta(outputs, k)

                    if ensemble_accum is None:
                        ensemble_accum = outputs
                    else:
                        ensemble_accum += outputs
                    count += 1

            avg_pred = ensemble_accum / count

            # Post-processing: Crop to original size
            img_id_str = str(img_id[0])
            orig_noisy = val_dataset.noisy_imgs[img_id_str]
            orig_h, orig_w = orig_noisy.shape

            pred_np = avg_pred.squeeze().cpu().numpy()  # (H_pad, W_pad)

            curr_h, curr_w = pred_np.shape
            pad_h = curr_h - orig_h
            pad_w = curr_w - orig_w
            pt = pad_h // 2
            pl = pad_w // 2

            final_pred = pred_np[pt : pt + orig_h, pl : pl + orig_w]
            final_clean = val_dataset.clean_imgs[img_id_str]

            y_pred.append(final_pred)
            y_true.append(final_clean)

            # Failure Analysis Data
            mse_img = np.mean((final_clean - final_pred) ** 2)
            rmse_img = np.sqrt(mse_img)
            errors.append(rmse_img)

            means.append(np.mean(orig_noisy))
            stds.append(np.std(orig_noisy))

    # Compute Global RMSE
    final_metric = calculate_rmse(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("Failure Analysis:")
    if len(errors) > 1:
        corr_mean = np.corrcoef(errors, means)[0, 1]
        corr_std = np.corrcoef(errors, stds)[0, 1]
        print(f"Correlation (Error vs Input Mean): {corr_mean:.4f}")
        print(f"Correlation (Error vs Input Std): {corr_std:.4f}")

    # 3. Submission
    THRESHOLD = 0.011870221132053216
    if final_metric < THRESHOLD:
        print("Metric threshold met. Generating submission...")
        predict_test_set()
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
