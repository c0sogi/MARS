import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.utils import set_seed, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import FastRepVGG
from library.engine import train_classifier, generate_submission

# --- Configuration ---
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 20
BATCH_SIZE = 64
PATIENCE = 5
SAVE_DIR = "./working/idea_32"
SUBMISSION_FILE = "./submission/submission.csv"


def run_validation_analysis(seeds, device):
    """
    Performs ensemble validation and failure analysis.
    """
    print("\n--- Starting Ensemble Validation & Failure Analysis ---")

    # Load validation data
    # Note: val_loader has shuffle=False, so order is deterministic
    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True)

    # 1. Collect Ground Truth and Image Meta-Features
    all_targets = []
    all_brightness = []
    all_contrast = []

    for images, labels in val_loader:
        # Store targets
        all_targets.append(labels.numpy())

        # Calculate meta-features for failure analysis
        # images is (B, C, H, W) in [0, 1] range
        imgs_np = images.numpy()

        # Brightness: Mean intensity
        b = np.mean(imgs_np, axis=(1, 2, 3))
        all_brightness.append(b)

        # Contrast: Standard deviation of intensity
        c = np.std(imgs_np, axis=(1, 2, 3))
        all_contrast.append(c)

    targets = np.concatenate(all_targets)
    brightness = np.concatenate(all_brightness)
    contrast = np.concatenate(all_contrast)

    # 2. Collect Predictions from all seeds
    all_seed_preds = []

    for seed in seeds:
        model_path = os.path.join(SAVE_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} missing. Skipping.")
            continue

        # Load Model
        model = FastRepVGG(num_classes=1, deploy=False)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Switch to Deploy Mode (Structural Re-parameterization)
        model.switch_to_deploy()
        model.to(device)
        model.eval()

        # Generate Predictions with TTA (Original + HFlip + VFlip)
        seed_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)

                # TTA 1: Original
                out1 = torch.sigmoid(model(images))
                # TTA 2: Horizontal Flip
                out2 = torch.sigmoid(model(torch.flip(images, [3])))
                # TTA 3: Vertical Flip
                out3 = torch.sigmoid(model(torch.flip(images, [2])))

                # Average
                avg_out = (out1 + out2 + out3) / 3.0
                seed_preds.append(avg_out.cpu().numpy())

        all_seed_preds.append(np.concatenate(seed_preds).flatten())

    if not all_seed_preds:
        print("No valid models found. Cannot perform validation.")
        return 0.0

    # Ensemble Averaging
    ensemble_preds = np.mean(all_seed_preds, axis=0)

    # 3. Calculate Final Metric
    final_auc = calculate_roc_auc(targets, ensemble_preds)
    print(f"Final Validation Metric: {final_auc:.10f}")

    # 4. Failure Analysis
    # Calculate absolute error
    errors = np.abs(targets - ensemble_preds)

    # Calculate correlations
    corr_b, _ = pearsonr(errors, brightness)
    corr_c, _ = pearsonr(errors, contrast)

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    print(f"Brightness: {corr_b:.4f}")
    print(f"Contrast:   {corr_c:.4f}")

    return final_auc


def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Phase 1: Training ---
    print("\n=== Phase 1: Training Ensemble ===")
    for seed in SEEDS:
        print(f"\nTraining Seed {seed}...")
        train_classifier(
            seed=seed,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            save_dir=SAVE_DIR,
        )

    # --- Phase 2: Validation & Analysis ---
    print("\n=== Phase 2: Validation & Analysis ===")
    val_metric = run_validation_analysis(SEEDS, device)

    # --- Phase 3: Submission ---
    print("\n=== Phase 3: Submission ===")
    # Note: The requirement "higher than 1.0" for AUC is mathematically impossible.
    # Assuming a standard threshold of 0.5 (random guessing).
    threshold = 0.5

    if val_metric > threshold:
        print(
            f"Validation metric ({val_metric:.4f}) meets threshold. Generating submission..."
        )
        generate_submission(seeds=SEEDS, save_dir=SAVE_DIR, output_file=SUBMISSION_FILE)
    else:
        print(f"Validation metric ({val_metric:.4f}) is too low. Skipping submission.")


if __name__ == "__main__":
    main()
