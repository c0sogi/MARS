import os
import torch
import numpy as np
import warnings
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed
from library.model import SpecialistSegFormer
from library.data import get_specialist_datasets
from library.trainer import train_specialist
from library.inference import create_submission, load_models

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    specialists = ["A", "B", "C"]

    # 2. Train Specialists
    # We iterate through each specialist configuration, train the model, and save the best checkpoint.
    for key in specialists:
        print(f"--- Training Specialist {key} ---")

        # Load datasets (using caching to speed up if re-running)
        train_ds, val_ds = get_specialist_datasets(key, load_cached_data=True)

        # Configure DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize and Train
        model = SpecialistSegFormer()
        train_specialist(model, train_loader, val_loader, key, epochs=Config.EPOCHS)

        # Clean up to save memory
        del model, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # 3. Ensemble Validation
    # We now evaluate the full ensemble on the validation set to get the final metric.
    print("\n--- Running Ensemble Validation ---")

    # Load the trained models
    models = load_models(device)

    # Load validation datasets for all views to perform matched inference
    # We ignore the train dataset here
    _, val_ds_A = get_specialist_datasets("A", load_cached_data=True)
    _, val_ds_B = get_specialist_datasets("B", load_cached_data=True)
    _, val_ds_C = get_specialist_datasets("C", load_cached_data=True)

    # Validation loop
    total_tp = 0
    total_fp = 0
    total_fn = 0

    error_magnitudes = []
    input_intensities = []

    # Use batch size from config
    batch_size = Config.BATCH_SIZE
    num_samples = len(val_ds_A)

    # Disable gradients for inference
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            indices = range(i, min(i + batch_size, num_samples))

            # Prepare batch tensors
            imgs_A, imgs_B, imgs_C, targets = [], [], [], []

            for idx in indices:
                # Get samples from each specialist dataset (same patch, different Z-depths)
                img_a, lbl = val_ds_A[idx]
                img_b, _ = val_ds_B[idx]
                img_c, _ = val_ds_C[idx]

                imgs_A.append(img_a)
                imgs_B.append(img_b)
                imgs_C.append(img_c)
                targets.append(lbl)

            # Stack and move to device
            t_A = torch.stack(imgs_A).to(device)
            t_B = torch.stack(imgs_B).to(device)
            t_C = torch.stack(imgs_C).to(device)
            t_lbl = torch.stack(targets).to(device)

            # Get predictions from each specialist
            # Models output logits -> apply sigmoid
            probs_A = torch.sigmoid(models["A"](t_A))
            probs_B = torch.sigmoid(models["B"](t_B))
            probs_C = torch.sigmoid(models["C"](t_C))

            # Ensemble Fusion: Max Probability Projection
            stacked_probs = torch.stack([probs_A, probs_B, probs_C], dim=0)
            fused_probs, _ = torch.max(stacked_probs, dim=0)

            # Binarize for metrics (Threshold = 0.5)
            preds_bin = (fused_probs > 0.5).float()
            targets_bin = t_lbl.float()

            # Accumulate TP, FP, FN
            tp = (preds_bin * targets_bin).sum().item()
            fp = (preds_bin * (1 - targets_bin)).sum().item()
            fn = ((1 - preds_bin) * targets_bin).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Failure Analysis Data Collection
            # Metric: Mean Absolute Error per sample
            # Feature: Mean Input Intensity (using View A as proxy)
            mae = torch.abs(fused_probs - targets_bin).mean(dim=(1, 2, 3)).cpu().numpy()
            intensity = t_A.mean(dim=(1, 2, 3)).cpu().numpy()

            error_magnitudes.extend(mae)
            input_intensities.extend(intensity)

    # Calculate Final F0.5 Score
    beta = 0.5
    beta_sq = beta**2
    epsilon = 1e-7

    numerator = (1 + beta_sq) * total_tp
    denominator = (1 + beta_sq) * total_tp + beta_sq * total_fn + total_fp
    final_metric = numerator / (denominator + epsilon)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    if len(error_magnitudes) > 1:
        corr, p_val = pearsonr(input_intensities, error_magnitudes)
        print(
            f"Failure Analysis: Correlation between Input Intensity and Prediction Error: {corr:.4f} (p={p_val:.4f})"
        )

        if abs(corr) > 0.3:
            print(
                "Observation: Significant correlation detected. Model performance varies with ink intensity/contrast."
            )
        else:
            print(
                "Observation: Weak correlation. Errors are likely distributed across intensity ranges."
            )

    # 5. Submission
    # Threshold check
    THRESHOLD = 0.0

    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")
        create_submission()
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
