import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.train import train_model
from library.inference import predict
from library.dataset import get_loaders
from library.model import DeepResUNet
from library.utils import set_seed, compute_salt_metric


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    # We modify the configuration to run a shorter but complete curriculum.
    # 75 epochs total allows for 3 cycles of 25 epochs each, fitting well within the time limit.
    Config.EPOCHS_PER_CYCLE = 25
    Config.TOTAL_EPOCHS = 75
    Config.CYCLE_1_END_EPOCH = 25  # Switch loss function after Cycle 1
    Config.SAVE_CYCLES = [2, 3]  # Save best models from Cycle 2 and 3

    # Setup directories and seeds
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("--- Starting Training Pipeline ---")
    # This function handles data loading, model initialization, and the training loop
    train_model()

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load Validation Data
    # We use load_cached_data=True to leverage the .npy files created during training
    _, val_loader, _ = get_loaders(load_cached_data=True)

    # Load Models for Ensemble (Cycle 2 and Cycle 3)
    # The strategy uses an ensemble of the best snapshots from the Lovasz optimization phase
    models = []
    for cycle in [2, 3]:
        path = os.path.join(Config.CHECKPOINT_DIR, f"best_cycle_{cycle}.pth")
        if os.path.exists(path):
            print(f"Loading checkpoint: {path}")
            m = DeepResUNet().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
        else:
            print(f"Warning: Checkpoint for Cycle {cycle} not found.")

    # Fallback to global best if cycle models are missing
    if not models:
        print("No cycle checkpoints found. Attempting to load global best model.")
        path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(path):
            m = DeepResUNet().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)

    if not models:
        print("Error: No trained models found. Exiting.")
        return

    # Load Metadata for Analysis (Depth 'z')
    df_val = pd.read_csv(Config.VAL_CSV)
    id_to_z = dict(zip(df_val["id"], df_val["z"]))

    val_maps = []
    val_ious = []
    val_depths = []
    val_coverages = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)  # Shape: (B, 1, H, W)

            # Ensemble Prediction
            avg_probs = torch.zeros_like(masks)

            for model in models:
                # Forward pass
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Test-Time Augmentation (Horizontal Flip)
                if Config.TTA_FLIP:
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped)
                    probs_flipped_back = torch.flip(probs_flipped, dims=[3])
                    probs = 0.5 * (probs + probs_flipped_back)

                avg_probs += probs

            # Average predictions
            avg_probs /= len(models)

            # Move to CPU for metric calculation
            probs_np = avg_probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Process batch
            for i in range(len(ids)):
                img_id = ids[i]
                p = probs_np[i, 0]
                t = masks_np[i, 0]

                # Binarize for metrics (Threshold 0.5)
                p_bin = (p > 0.5).astype(np.uint8)
                t_bin = (t > 0.5).astype(np.uint8)

                # 1. Calculate Competition Metric (mAP over thresholds)
                score = compute_salt_metric(p_bin, t_bin)
                val_maps.append(score)

                # 2. Calculate IoU (for failure analysis)
                intersection = np.logical_and(p_bin, t_bin).sum()
                union = np.logical_or(p_bin, t_bin).sum()

                if union == 0:
                    iou = 1.0 if t_bin.sum() == 0 else 0.0
                else:
                    iou = intersection / union
                val_ious.append(iou)

                # 3. Collect Metadata
                z = id_to_z.get(img_id, 0)
                val_depths.append(z)

                # Salt Coverage (Ground Truth)
                cov = t_bin.sum() / t_bin.size
                val_coverages.append(cov)

    # Compute Final Metric
    final_metric = np.mean(val_maps)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis Results
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis Report ---")
    errors = 1.0 - np.array(val_ious)
    depths = np.array(val_depths)
    coverages = np.array(val_coverages)

    # Correlation: Error vs Depth
    if np.std(errors) > 0 and np.std(depths) > 0:
        corr_depth, _ = pearsonr(errors, depths)
        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    else:
        print("Correlation (Error vs Depth): N/A (Constant values)")

    # Correlation: Error vs Salt Coverage
    if np.std(errors) > 0 and np.std(coverages) > 0:
        corr_cov, _ = pearsonr(errors, coverages)
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")
    else:
        print("Correlation (Error vs Salt Coverage): N/A (Constant values)")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.833
    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.4f}) exceeds threshold ({THRESHOLD})."
        )
        print("Generating submission for test set...")
        # predict() uses the Checkpoints saved in Config.CHECKPOINT_DIR
        # It automatically ensembles Cycle 2 and Cycle 3 if present.
        predict()
    else:
        print(
            f"\nValidation metric ({final_metric:.4f}) does NOT exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
