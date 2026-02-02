import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.trainer import Trainer
from library.model import DeepResUNet
from library.dataset import get_dataloaders
from library.utils import rle_encode, compute_map_score, set_seed


def run_inference(checkpoints, loader, device, is_test=False):
    """
    Runs inference using Snapshot Ensemble and TTA.
    Returns:
        preds: (N, H, W) numpy array of probabilities (cropped to 101x101)
        targets: (N, H, W) numpy array of ground truth (if not is_test)
        ids: List of IDs
    """
    # 1. Initialize Model
    model = DeepResUNet().to(device)
    model.eval()

    # 2. Prepare Accumulators
    # We need to accumulate predictions across all checkpoints
    # Since dataset size might be large, we process batch by batch and accumulate in a list
    # But we need to average over checkpoints first.
    # Strategy: Iterate over checkpoints, for each checkpoint iterate over loader.
    # To save memory, we can't store all epoch preds for all models.
    # Better: Iterate loader, inside iterate checkpoints.

    all_preds = []
    all_targets = []
    all_ids = []

    # Crop indices
    start_idx = (Config.IMG_HEIGHT - Config.ORIG_HEIGHT) // 2
    end_idx = start_idx + Config.ORIG_HEIGHT

    print(f"Starting inference on {'Test' if is_test else 'Validation'} set...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if is_test:
                images, depths, ids = batch_data
                masks = None
            else:
                images, masks, depths, ids = batch_data

            images = images.to(device)
            depths = depths.to(device)

            batch_ensemble_preds = None

            # Iterate over each checkpoint for the ensemble
            for ckpt_path in checkpoints:
                # Load weights
                # Note: Loading state_dict inside the loop is slightly inefficient but
                # safer for memory than keeping multiple models.
                # Given the small model size, we could load all models, but let's stick to serial.
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)

                # Forward Pass (Original)
                logits = model(images, depths)
                preds = torch.sigmoid(logits)

                # Forward Pass (TTA - Horizontal Flip)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped, depths)
                preds_flipped = torch.sigmoid(logits_flipped)
                preds_flipped = torch.flip(preds_flipped, dims=[3])

                # Average TTA
                preds_avg = (preds + preds_flipped) / 2.0

                if batch_ensemble_preds is None:
                    batch_ensemble_preds = preds_avg
                else:
                    batch_ensemble_preds += preds_avg

            # Average over checkpoints
            batch_ensemble_preds /= len(checkpoints)

            # Crop to original size
            preds_cropped = batch_ensemble_preds[
                :, :, start_idx:end_idx, start_idx:end_idx
            ]

            # Store results
            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
            all_preds.append(preds_cropped.squeeze(1).cpu().numpy())
            all_ids.extend(ids)

            if masks is not None:
                masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]
                all_targets.append(masks_cropped.squeeze(1).cpu().numpy())

    # Concatenate all batches
    final_preds = np.concatenate(all_preds, axis=0)
    final_targets = np.concatenate(all_targets, axis=0) if all_targets else None

    return final_preds, final_targets, np.array(all_ids)


def analyze_failures(preds, targets, ids):
    """
    Performs failure analysis by correlating error with metadata.
    """
    # 1. Load Metadata
    df_val = pd.read_csv(Config.VAL_CSV)

    # Map IDs to metadata
    id_to_meta = df_val.set_index("id")[["z", "coverage"]].to_dict("index")

    # 2. Calculate Per-Image mAP
    # compute_map_score computes mean over batch. We need per-image.
    # We'll implement a simplified version here or reuse logic.
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Binarize preds (0.5 threshold for IoU calculation base)
    preds_bin = (preds > 0.5).astype(np.uint8)
    targets_bin = targets.astype(np.uint8)

    # Flatten
    N = preds.shape[0]
    preds_flat = preds_bin.reshape(N, -1)
    targets_flat = targets_bin.reshape(N, -1)

    intersection = (preds_flat * targets_flat).sum(axis=1)
    union = (preds_flat + targets_flat).astype(bool).astype(int).sum(axis=1)

    ious = np.ones(N, dtype=float)
    mask_union = union > 0
    ious[mask_union] = intersection[mask_union] / union[mask_union]

    # Calculate mAP per image
    # matches: (N, T)
    matches = ious[:, None] > thresholds[None, :]
    per_image_map = matches.mean(axis=1)

    # 3. Correlate
    depths = []
    coverages = []
    maps = []

    for i, img_id in enumerate(ids):
        if img_id in id_to_meta:
            meta = id_to_meta[img_id]
            depths.append(meta["z"])
            coverages.append(meta["coverage"])
            maps.append(per_image_map[i])

    if not maps:
        print("Could not match IDs to metadata for analysis.")
        return

    # Calculate Correlations
    corr_depth, _ = pearsonr(depths, maps)
    corr_cov, _ = pearsonr(coverages, maps)

    print("\n--- Failure Analysis ---")
    print(f"Correlation (Depth vs mAP): {corr_depth:.4f}")
    print(f"Correlation (Salt Coverage vs mAP): {corr_cov:.4f}")

    # Additional: Error vs Image Brightness (if we had image stats easily available)
    # We can infer that low coverage (empty masks) might be easier or harder.
    print(f"Average mAP: {np.mean(maps):.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Train
    # The Trainer handles the entire training loop, curriculum, and checkpointing
    trainer = Trainer()
    trainer.fit()

    # 3. Identify Checkpoints for Ensemble
    # We look for best_cycle_2.pth and best_cycle_3.pth
    ensemble_ckpts = []
    for cycle in Config.SNAPSHOT_CYCLES:
        path = os.path.join(Config.CHECKPOINT_DIR, f"best_cycle_{cycle}.pth")
        if os.path.exists(path):
            ensemble_ckpts.append(path)

    # Fallback if cycles didn't complete or save
    if not ensemble_ckpts:
        print("Warning: Cycle checkpoints not found. Falling back to best_model.pth")
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            ensemble_ckpts.append(best_model_path)
        else:
            print("Error: No checkpoints found!")
            return

    print(
        f"Using checkpoints for ensemble: {[os.path.basename(p) for p in ensemble_ckpts]}"
    )

    # 4. Load DataLoaders
    # We need val and test loaders. Trainer already loaded them but stored in self.
    # We can reload or access from trainer. Let's reload to be clean and independent.
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 5. Validation Inference (Ensemble)
    val_preds, val_targets, val_ids = run_inference(
        ensemble_ckpts, val_loader, device, is_test=False
    )

    # 6. Compute Final Metric
    # compute_map_score expects (N, H, W)
    final_val_metric = compute_map_score(val_preds, val_targets)
    print(f"Final Validation Metric: {final_val_metric:.10f}")

    # 7. Failure Analysis
    analyze_failures(val_preds, val_targets, val_ids)

    # 8. Submission
    if final_val_metric > 0.833:
        print("\nValidation metric passed threshold. Generating submission...")

        # Inference on Test Set
        test_preds, _, test_ids = run_inference(
            ensemble_ckpts, test_loader, device, is_test=True
        )

        # Thresholding
        # We use 0.5 as the decision boundary for the binary mask
        test_masks = (test_preds > 0.5).astype(np.uint8)

        # RLE Encoding
        submission_rows = []
        for i, img_id in enumerate(test_ids):
            mask = test_masks[i]
            rle = rle_encode(mask)
            submission_rows.append({"id": img_id, "rle_mask": rle})

        # Create DataFrame
        df_sub = pd.DataFrame(submission_rows)

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")

    else:
        print(
            f"\nFinal Validation Metric ({final_val_metric:.4f}) is below threshold (0.833). Submission skipped."
        )


if __name__ == "__main__":
    main()
