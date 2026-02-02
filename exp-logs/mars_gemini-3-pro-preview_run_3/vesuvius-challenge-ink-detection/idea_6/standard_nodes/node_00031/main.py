import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

# Ensure the current directory is in the path for module imports
sys.path.append(str(Path.cwd()))

from library.config import Config
from library.utils import seed_everything, fbeta_score, load_volume
from library.model import SGDN
from library.train import train, validate
from library.inference import inference
from library.data import get_loaders


def run_pipeline():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Configure for Fast Baseline
    # Limit epochs to ensure execution finishes quickly (within 2 hours)
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 16

    # Set submission path as required
    Config.SUBMISSION_PATH = Path("./submission/submission.csv")
    if Config.SUBMISSION_PATH.parent != Path("."):
        Config.SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 2. Training
    # train() uses the modified Config values
    train()

    # 3. Evaluation on Validation Set
    # Reload the best model to compute the final metric
    model = SGDN().to(device)
    best_model_path = Config.WORKING_DIR / "best_model.pth"

    if not best_model_path.exists():
        print("Error: best_model.pth not found after training.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load the optimized threshold found during training
    threshold_path = Config.WORKING_DIR / "threshold.txt"
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            best_threshold = float(f.read().strip())
    else:
        best_threshold = 0.5

    # Get validation data loader
    _, val_loader = get_loaders()

    # Run inference on validation set
    # validate() returns flattened arrays of probabilities and binary targets for valid pixels
    val_preds, val_targets = validate(model, val_loader, device)

    # Compute Final Metric (F0.5)
    final_metric = fbeta_score(
        val_preds, val_targets, beta=0.5, threshold=best_threshold
    )
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Correlate prediction error with input pixel intensity

    # Load validation fragment IDs
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_frag_ids = df_val["fragment_id"].astype(str).unique()

    all_intensities = []

    # Extract intensities in the same order as validation predictions
    for fid in val_frag_ids:
        # Load volume (Z, H, W) and mask
        vol, mask, _ = load_volume(fid, split="val", load_cached_data=True)

        # Compute mean intensity across Z-depth (feature proxy)
        mean_intensity = np.mean(vol, axis=0)

        # Normalize (consistent with model input)
        mean_intensity = (mean_intensity - Config.PIXEL_MEAN) / Config.PIXEL_STD

        # Flatten valid pixels
        valid_mask = mask > 0
        all_intensities.append(mean_intensity[valid_mask])

    if all_intensities:
        all_intensities = np.concatenate(all_intensities)

        # Calculate Error: Absolute difference between predicted probability and binary target
        errors = np.abs(val_preds - val_targets)

        if len(errors) == len(all_intensities):
            corr, _ = pearsonr(errors, all_intensities)
            print(f"Error vs Intensity Correlation: {corr}")
        else:
            print(
                f"Shape mismatch in failure analysis: {len(errors)} vs {len(all_intensities)}"
            )
    else:
        print("No validation data available for failure analysis.")

    # 5. Submission
    # Only generate submission if metric exceeds the specified baseline
    TARGET_METRIC = 0.39266693592071533

    if final_metric > TARGET_METRIC:
        # inference() generates predictions for test set and saves to Config.SUBMISSION_PATH
        inference()
    else:
        print(
            f"Metric ({final_metric}) did not exceed target ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
