import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_weighted_log_loss_score, load_checkpoint
from library.train import run_training
from library.inference import predict_test_set
from library.data import get_dataloaders
from library.model import CervicalFractureNet

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    # This runs the training loop and saves the best model to ./working/idea_10/best_model.pth
    print("\n=== Starting Training Phase ===")
    run_training()

    # 3. Validation & Failure Analysis
    print("\n=== Starting Validation & Analysis Phase ===")

    # Load the best model
    model = CervicalFractureNet()
    model.to(device)

    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL ERROR: Checkpoint not found at {checkpoint_path}")
        return

    # Load weights
    load_checkpoint(checkpoint_path, model, device=Config.DEVICE)
    model.eval()

    # Get Validation Data
    # load_cached_data=True uses the parquet cache for speed
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    all_preds = []
    all_targets = []
    study_ids = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["study_labels"].to(device)
            ids = batch["study_id"]

            # Forward pass
            outputs = model(images)
            logits = outputs["study_logits"]
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            study_ids.extend(ids)

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 4. Calculate Final Metric
    # Weights: C1-C7 (1.0), Patient Overall (7.0)
    final_metric = get_weighted_log_loss_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate loss per study to correlate with metadata
    # Weights for loss calculation
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    # Clip predictions for stability
    epsilon = 1e-15
    y_pred_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    # Calculate weighted log loss per sample
    # Formula: -w * [y*log(p) + (1-y)*log(1-p)]
    loss_matrix = -(
        all_targets * np.log(y_pred_clipped)
        + (1 - all_targets) * np.log(1 - y_pred_clipped)
    )
    weighted_loss_matrix = loss_matrix * weights

    # Average loss per study (across the 8 columns)
    study_losses = np.mean(weighted_loss_matrix, axis=1)

    # Extract Feature: Slice Count (Z-depth)
    # We read the validation metadata to get paths and count files
    val_meta_path = Config.VAL_METADATA_PATH
    slice_counts = []

    if os.path.exists(val_meta_path):
        val_df = pd.read_csv(val_meta_path)
        # Create a map for quick lookup
        # The val_loader order might differ if shuffle was True, but val_loader shuffle is False.
        # However, to be safe, we map study_id -> slice_count
        study_slice_map = {}

        for _, row in val_df.iterrows():
            uid = row["StudyInstanceUID"]
            rel_path = row["image_path"]
            full_path = os.path.join(Config.INPUT_ROOT, rel_path)

            count = 0
            if os.path.exists(full_path):
                # Fast count of .dcm files
                try:
                    count = len(
                        [
                            name
                            for name in os.listdir(full_path)
                            if name.endswith(".dcm")
                        ]
                    )
                except OSError:
                    count = 0
            study_slice_map[uid] = count

        # Align slice counts with the prediction order
        for uid in study_ids:
            slice_counts.append(study_slice_map.get(uid, 0))

        slice_counts = np.array(slice_counts)

        # Calculate Correlation
        if len(study_losses) > 1 and np.std(slice_counts) > 0:
            # Pearson correlation
            correlation_matrix = np.corrcoef(study_losses, slice_counts)
            correlation = correlation_matrix[0, 1]
            print(
                f"Correlation between Error Magnitude and Slice Count: {correlation:.4f}"
            )
        else:
            print("Correlation could not be computed (insufficient data or variance).")
    else:
        print("Validation metadata not found, skipping feature extraction.")

    # 6. Submission Generation
    THRESHOLD = 0.15364714496434773

    print("\n=== Submission Check ===")
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        predict_test_set()
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
