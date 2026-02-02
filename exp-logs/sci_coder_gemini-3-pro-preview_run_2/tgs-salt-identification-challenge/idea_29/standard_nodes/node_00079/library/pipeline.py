import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from sklearn.model_selection import KFold
import copy
import cv2

from library.config import Config, seed_everything
from library.utils import unpad_image, pad_image, rle_encode, calc_map
from library.dataset import process_data, SaltDataset, get_transforms, get_loaders
from library.models import SaltModel
from library.training import train_model, generate_submission
from library.losses import LovaszHingeLoss

# Ensure reproducibility
seed_everything(Config.SEED)


def run_cv_training(debug=False):
    """
    Runs 5-Fold Cross-Validation Training on the SaltModel.
    """
    print("=" * 50)
    print("Running 5-Fold CV Training")
    print("=" * 50)

    # Load All Data
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Process data (force reload to ensure clean state)
    t_imgs, t_masks, t_depths, t_ids = process_data(
        train_df, "train", Config.CACHE_DIR, load_cached_data=False
    )
    v_imgs, v_masks, v_depths, v_ids = process_data(
        val_df, "val", Config.CACHE_DIR, load_cached_data=False
    )

    # Combine for CV
    all_images = np.concatenate([t_imgs, v_imgs], axis=0)
    all_masks = np.concatenate([t_masks, v_masks], axis=0)
    all_depths = np.concatenate([t_depths, v_depths], axis=0)
    all_ids = np.concatenate([t_ids, v_ids], axis=0)

    # Normalize Depths
    d_mean = np.mean(all_depths)
    d_std = np.std(all_depths) + 1e-8
    all_depths_norm = (all_depths - d_mean) / d_std

    # Save stats for test inference
    stats_path = os.path.join(Config.CACHE_DIR, "depth_stats.csv")
    pd.DataFrame({"mean": [d_mean], "std": [d_std]}).to_csv(stats_path, index=False)

    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    oof_preds_dict = {}  # Map id -> (pred_prob, mask)
    valid_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        if debug and fold > 0:
            break

        # Datasets
        train_ds = SaltDataset(
            all_images[train_idx],
            all_depths_norm[train_idx],
            all_ids[train_idx],
            all_masks[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = SaltDataset(
            all_images[val_idx],
            all_depths_norm[val_idx],
            all_ids[val_idx],
            all_masks[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model
        model = SaltModel(num_classes=1).to(Config.DEVICE)

        # Train
        model, best_oof_preds, best_oof_masks = train_model(
            model, train_loader, val_loader, Config.DEVICE, Config, fold_idx=fold
        )

        # Gating Check (Cite Lesson 00067)
        # Calculate mAP for this fold
        best_thresh = 0.5
        best_map = 0.0
        thresholds = np.arange(0.3, 0.72, 0.02)
        for t in thresholds:
            score = calc_map((best_oof_preds > t).astype(np.uint8), best_oof_masks)
            if score > best_map:
                best_map = score
                best_thresh = t

        print(f"Fold {fold+1} Best mAP: {best_map:.6f} at Threshold {best_thresh:.2f}")

        if best_map >= 0.70:
            valid_model_paths.append(
                os.path.join(Config.CACHE_DIR, f"best_model_fold{fold}.pth")
            )
            # Store OOF preds
            fold_ids = all_ids[val_idx]
            for i, uid in enumerate(fold_ids):
                oof_preds_dict[uid] = (best_oof_preds[i], best_oof_masks[i])
        else:
            print(f"Fold {fold+1} discarded (mAP < 0.70)")

    # Optimize Global Threshold on OOF
    print("\nOptimizing Global Threshold on OOF Predictions...")
    if not oof_preds_dict:
        print("No valid OOF predictions.")
        return [], 0.5, 0.0

    all_oof_preds = []
    all_oof_masks = []
    for uid in oof_preds_dict:
        p, m = oof_preds_dict[uid]
        all_oof_preds.append(p)
        all_oof_masks.append(m)

    all_oof_preds = np.array(all_oof_preds)
    all_oof_masks = np.array(all_oof_masks)

    best_global_map = 0.0
    best_global_thresh = 0.5
    for t in np.arange(0.3, 0.72, 0.02):
        score = calc_map((all_oof_preds > t).astype(np.uint8), all_oof_masks)
        if score > best_global_map:
            best_global_map = score
            best_global_thresh = t

    print(
        f"Global OOF mAP: {best_global_map:.6f} at Threshold {best_global_thresh:.4f}"
    )

    # Failure Analysis (Correlation Error vs Depth)
    # Cite Lesson 00029: Check correlation to ensure depth is being used effectively.
    binary_oof = (all_oof_preds > best_global_thresh).astype(np.uint8)
    intersection = np.sum(binary_oof & all_oof_masks, axis=(1, 2))
    union = np.sum(binary_oof | all_oof_masks, axis=(1, 2))

    ious = np.ones_like(intersection, dtype=np.float32)
    valid_mask = union > 0
    ious[valid_mask] = intersection[valid_mask] / union[valid_mask]
    errors = 1.0 - ious

    # We need depths corresponding to OOF. The OOF loop iterated over dict keys (ids).
    # We need to fetch depths for these IDs.
    # Map ID -> Depth
    id_to_depth = {uid: d for uid, d in zip(all_ids, all_depths)}  # Raw depths
    oof_depths = np.array([id_to_depth[uid] for uid in oof_preds_dict.keys()])

    if np.std(errors) > 0 and np.std(oof_depths) > 0:
        correlation = np.corrcoef(errors, oof_depths)[0, 1]
    else:
        correlation = 0.0

    print(f"Failure Analysis - Correlation (Error vs Depth): {correlation:.10f}")

    return valid_model_paths, best_global_thresh, best_global_map


def generate_ensemble_submission(model_paths, threshold):
    """
    Generates submission by ensembling predictions from multiple models.
    """
    print("Generating Ensemble Submission...")

    # Load Test Data
    test_df = pd.read_csv(Config.TEST_CSV)
    test_imgs, _, test_depths_raw, test_ids = process_data(
        test_df, "test", Config.CACHE_DIR, load_cached_data=False
    )

    # Normalize Depths
    stats_path = os.path.join(Config.CACHE_DIR, "depth_stats.csv")
    stats = pd.read_csv(stats_path)
    d_mean, d_std = stats["mean"][0], stats["std"][0]
    test_depths_norm = (test_depths_raw - d_mean) / d_std

    test_ds = SaltDataset(
        test_imgs,
        test_depths_norm,
        test_ids,
        masks=None,
        transform=get_transforms("test"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Models
    models = []
    for path in model_paths:
        m = SaltModel(num_classes=1).to(Config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        m.eval()
        models.append(m)

    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(Config.DEVICE).float()
            depths = batch["depth"].to(Config.DEVICE).float()
            ids = batch["id"]

            batch_probs = torch.zeros(
                (images.size(0), Config.IMG_SIZE, Config.IMG_SIZE), device=Config.DEVICE
            )

            for model in models:
                logits = model(images, depths)
                probs = torch.sigmoid(logits).squeeze(1)

                # TTA: Flip
                logits_flip = model(torch.flip(images, [3]), depths)
                probs_flip = torch.flip(
                    torch.sigmoid(logits_flip).squeeze(1), [2]
                )  # Flip back W dim (2 since N,H,W)

                batch_probs += (probs + probs_flip) / 2.0

            batch_probs /= len(models)
            batch_probs = batch_probs.cpu().numpy()

            for i, img_id in enumerate(ids):
                p_un = unpad_image(batch_probs[i])
                mask = (p_un > threshold).astype(np.uint8)
                rle = rle_encode(mask)
                submission_data.append([img_id, rle])

    df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
