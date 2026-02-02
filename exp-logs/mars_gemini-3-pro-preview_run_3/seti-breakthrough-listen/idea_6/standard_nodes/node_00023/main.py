import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.engine import run_training, predict
from library.dataset import get_train_val_loaders
from library.model import SiameseDifferenceNet
from library.utils import get_score, seed_everything


def main():
    # --- 1. Configuration ---
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # --- 2. Training Phase ---
    print(f"Starting training on full dataset for {Config.EPOCHS} epochs...")
    run_training(
        debug=Config.DEBUG,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.BEST_MODEL_PATH,
    )

    # --- 3. Full Validation & Failure Analysis ---
    print("\nStarting Full Validation and Failure Analysis...")

    device = Config.DEVICE

    # Load the best model
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model file not found.")
        return

    model = SiameseDifferenceNet()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load FULL validation set (debug=False ensures we get all validation data)
    _, val_loader = get_train_val_loaders(debug=False)

    all_targets = []
    all_preds = []
    meta_stats = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for on_imgs, off_imgs, targets in val_loader:
            on_imgs = on_imgs.to(device)
            off_imgs = off_imgs.to(device)

            # Inference
            # Use mixed precision for speed if available, though not strictly necessary for inference
            logits = model(on_imgs, off_imgs).squeeze(1)
            probs = torch.sigmoid(logits)

            # Collect results
            preds_np = probs.cpu().numpy()
            targets_np = targets.numpy()
            all_preds.extend(preds_np)
            all_targets.extend(targets_np)

            # --- Failure Analysis Data Collection ---
            # Move data to CPU for stat calculation
            on_np = on_imgs.cpu().numpy()
            off_np = off_imgs.cpu().numpy()

            for i in range(len(targets_np)):
                # Calculate simple signal statistics
                on_sample = on_np[i]
                off_sample = off_np[i]

                stats = {
                    "on_mean": np.mean(on_sample),
                    "on_max": np.max(on_sample),
                    "on_std": np.std(on_sample),
                    "off_mean": np.mean(off_sample),
                    "off_max": np.max(off_sample),
                    "diff_mean": np.mean(on_sample) - np.mean(off_sample),
                    "target": targets_np[i],
                    "pred": preds_np[i],
                    "error": np.abs(targets_np[i] - preds_np[i]),
                }
                meta_stats.append(stats)

    # Calculate Final Metric
    final_auc = get_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # --- 4. Failure Analysis Report ---
    print("\nFailure Analysis (Correlation with Prediction Error):")
    df_stats = pd.DataFrame(meta_stats)

    # Calculate correlation of features with the absolute error
    if not df_stats.empty:
        correlations = (
            df_stats.corr()["error"]
            .drop(["error", "target", "pred"])
            .sort_values(ascending=False)
        )
        print(correlations)
    else:
        print("No validation stats collected.")

    # --- 5. Submission Generation ---
    threshold = 0.7770832449065452
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        predict(model_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
