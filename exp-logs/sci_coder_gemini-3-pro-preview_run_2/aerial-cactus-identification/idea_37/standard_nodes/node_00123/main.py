import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.train import train_seed
from library.inference import generate_submission
from library.model_components import UltraWideSERepNeXt
from library.dataset import get_loaders


def main():
    # --- 1. Configuration Setup ---
    # Using settings from Config (optimized for 32x32 resolution)
    Config.DEBUG = False

    print("========================================")
    print("      STARTING PIPELINE EXECUTION       ")
    print("========================================")
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # --- 2. Training Phase ---
    print("\n--- Phase 1: Training Ensemble ---")
    for seed in Config.SEEDS:
        train_seed(seed)

    # --- 3. Validation & Failure Analysis ---
    print("\n--- Phase 2: Validation & Failure Analysis ---")
    val_auc = validate_and_analyze()

    # --- 4. Submission Phase ---
    print("\n--- Phase 3: Submission Generation ---")
    # The requirement "higher than 1.0" for AUC is impossible (max AUC is 1.0).
    # Assuming the intention was "better than random guessing (0.5)".
    if val_auc > 0.5:
        generate_submission()
    else:
        print(f"Validation AUC ({val_auc}) is too low. Skipping submission.")


def validate_and_analyze():
    """
    Performs ensemble inference on the validation set, computes the final metric,
    and analyzes failure modes by correlating errors with image features.
    """
    device = torch.device(Config.DEVICE)

    # 1. Load Validation Data
    # We only need the validation loader here
    _, val_loader, _ = get_loaders(load_cached_data=True)

    # 2. Load Ensemble Models
    models = []
    for seed in Config.SEEDS:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for seed {seed} missing. Skipping.")
            continue

        model = UltraWideSERepNeXt(num_classes=Config.NUM_CLASSES, deploy=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        model.switch_to_deploy()  # Optimize for inference
        models.append(model)

    if not models:
        print("Error: No models available for validation.")
        return 0.0

    # 3. Inference on Validation Set
    all_targets = []
    all_preds = []

    # Lists to store image stats for failure analysis
    feat_brightness = []
    feat_contrast = []
    feat_r_mean = []
    feat_g_mean = []
    feat_b_mean = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # --- Ensemble Prediction with TTA ---
            # TTA: Original, H-Flip, V-Flip
            imgs_orig = images
            imgs_h = torch.flip(images, dims=[3])
            imgs_v = torch.flip(images, dims=[2])

            batch_preds_sum = torch.zeros(batch_size, device=device)

            for model in models:
                p_orig = torch.sigmoid(model(imgs_orig)).view(-1)
                p_h = torch.sigmoid(model(imgs_h)).view(-1)
                p_v = torch.sigmoid(model(imgs_v)).view(-1)
                batch_preds_sum += p_orig + p_h + p_v

            # Average over (Models * Views)
            batch_preds_avg = batch_preds_sum / (len(models) * 3)

            all_preds.extend(batch_preds_avg.cpu().numpy())
            all_targets.extend(targets.numpy())

            # --- Feature Extraction for Failure Analysis ---
            # Images are (B, C, H, W) in [0, 1]
            # Calculate stats per image in the batch
            # Move to CPU for stat calculation
            imgs_np = images.cpu().numpy()  # (B, 3, 32, 32)

            for i in range(batch_size):
                img = imgs_np[i]  # (3, 32, 32)

                # Brightness: Mean of all pixels
                feat_brightness.append(np.mean(img))

                # Contrast: Std of all pixels
                feat_contrast.append(np.std(img))

                # Channel Means
                feat_r_mean.append(np.mean(img[0]))
                feat_g_mean.append(np.mean(img[1]))
                feat_b_mean.append(np.mean(img[2]))

    # 4. Compute Metric
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)

    # Error magnitude
    errors = np.abs(targets_arr - preds_arr)

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "brightness": feat_brightness,
            "contrast": feat_contrast,
            "red_mean": feat_r_mean,
            "green_mean": feat_g_mean,
            "blue_mean": feat_b_mean,
        }
    )

    print("Correlation between Prediction Error and Image Features:")
    features = ["brightness", "contrast", "red_mean", "green_mean", "blue_mean"]
    for feat in features:
        corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
        print(f"{feat.ljust(12)}: {corr:.4f}")

    return final_auc


if __name__ == "__main__":
    main()
