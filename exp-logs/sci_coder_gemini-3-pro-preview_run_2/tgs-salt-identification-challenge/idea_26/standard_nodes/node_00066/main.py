import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.dataset import SaltDataset
from library.model import ResNet34WideLinkNet
from library.engine import train_model, validate, predict_and_submit
from library.utils import unpad_image, calc_iou

# -------------------------------------------------------------------------
# Helper Classes & Functions
# -------------------------------------------------------------------------


class SafeSaltDataset(SaltDataset):
    """
    Subclass of SaltDataset that handles missing depths (NaNs) in the test set
    by filling them with the mean of valid depths. This prevents NaN propagation
    during Multi-Task Loss calculation in Stage 3.
    """

    def _load_depths_from_disk(self):
        # Load raw depths
        vals = self.df["z"].values.astype(np.float32)

        # Check for NaNs
        mask = np.isnan(vals)
        if mask.any():
            # Calculate mean of valid values
            valid_mean = np.nanmean(vals)
            if np.isnan(valid_mean):
                valid_mean = 0.0

            # Fill NaNs
            vals[mask] = valid_mean

        return vals


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Config.EPOCHS is already set to 40 in config.py
    PATIENCE = 10  # Increased patience for longer training

    # Ensure output directories exist
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 1: 5-Fold Supervised Training (Ensemble)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STAGE 1: 5-Fold Supervised Training (Ensemble)")
    print("=" * 40)

    # Load full training set
    full_train_ds = SaltDataset(mode="train", load_cached_data=True)

    # K-Fold Split
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    ensemble_models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_ds)):
        print(f"\n--- Training Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Create Subsets
        train_sub = Subset(full_train_ds, train_idx)
        val_sub = Subset(full_train_ds, val_idx)

        # Dataloaders
        train_loader = DataLoader(
            train_sub,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = ResNet34WideLinkNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Train
        save_path = os.path.join(Config.CHECKPOINTS_DIR, f"fold_{fold}_best.pth")
        best_threshold = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            epochs=Config.EPOCHS,
            patience=PATIENCE,
            save_path=save_path,
        )

        # Load best model for this fold to get final metric
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, best_map, _, _ = validate(model, val_loader, device, return_probs=True)

        print(
            f"Fold {fold+1} Best mAP: {best_map:.6f} at Threshold {best_threshold:.4f}"
        )
        fold_metrics.append(best_map)
        ensemble_models.append((save_path, best_threshold))

        # Clean up
        del model, optimizer, train_loader, val_loader, train_sub, val_sub
        torch.cuda.empty_cache()

    avg_map = np.mean(fold_metrics)
    print(f"\nAverage CV mAP: {avg_map:.6f}")

    # -------------------------------------------------------------------------
    # Validation & Failure Analysis (On Hold-out Val Set)
    # -------------------------------------------------------------------------
    # Note: In 5-Fold CV, we don't have a separate hold-out set if we used full_train_ds.
    # The 'val.csv' in metadata was a fixed split.
    # To be consistent with previous logic, we should probably check performance on the fixed val set
    # OR rely on the CV score.
    # Let's perform analysis on the fixed Validation Set using the Ensemble.

    print("\n" + "=" * 40)
    print(" Validation & Failure Analysis (Ensemble on Fixed Val Set)")
    print("=" * 40)

    val_ds = SaltDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference on Val Set
    accumulated_probs = {}

    for fold_idx, (model_path, _) in enumerate(ensemble_models):
        model = ResNet34WideLinkNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                depths = batch["depth"].to(device)
                ids = batch["id"]

                # TTA Flip
                logits = model(images, depths)
                probs = torch.sigmoid(logits)

                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, dims=[3])

                probs_avg = (probs + probs_flip) / 2.0
                probs_np = probs_avg.cpu().numpy()

                for i, img_id in enumerate(ids):
                    p = probs_np[i].squeeze()
                    p_unpad = unpad_image(p, (Config.ORIG_HEIGHT, Config.ORIG_WIDTH))

                    if img_id not in accumulated_probs:
                        accumulated_probs[img_id] = np.zeros_like(p_unpad)
                    accumulated_probs[img_id] += p_unpad

        del model
        torch.cuda.empty_cache()

    # Average and Optimize Threshold
    all_probs = []
    all_masks = []
    val_ids = val_ds.ids
    val_masks_dict = {id: mask for id, mask in zip(val_ds.ids, val_ds.masks)}

    for img_id in val_ids:
        avg_prob = accumulated_probs[img_id] / Config.N_FOLDS
        all_probs.append(avg_prob)
        all_masks.append(val_masks_dict[img_id])

    best_thresh, final_val_map = optimize_threshold(all_probs, all_masks)
    print(f"Final Ensemble Validation Metric: {final_val_map:.10f}")

    # Failure Analysis
    val_df = val_ds.df
    ious = []
    depths = []
    coverages = []

    # Map IDs to metadata
    id_to_depth = dict(zip(val_df.id, val_df.z))
    id_to_cov = dict(zip(val_df.id, val_df.salt_coverage))

    for i, img_id in enumerate(val_ids):
        prob = all_probs[i]
        mask = all_masks[i]
        pred_bin = (prob > best_thresh).astype(np.uint8)
        iou = calc_iou(pred_bin, mask)
        ious.append(iou)
        depths.append(id_to_depth[img_id])
        coverages.append(id_to_cov[img_id])

    ious = np.array(ious)
    depths = np.array(depths)
    coverages = np.array(coverages)

    if len(ious) > 1:
        corr_depth, _ = pearsonr(depths, ious)
        corr_cov, _ = pearsonr(coverages, ious)
    else:
        corr_depth, corr_cov = 0.0, 0.0

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (IoU vs Depth): {corr_depth:.4f}")
    print(f"Correlation (IoU vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    if final_val_map > 0.7985:
        print("\n=== Generating Submission (Ensemble) ===")

        test_ds = SaltDataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_probs = {}

        for fold_idx, (model_path, _) in enumerate(ensemble_models):
            print(f"Predicting Test Set with Fold {fold_idx+1}...")
            model = ResNet34WideLinkNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                for batch in test_loader:
                    images = batch["image"].to(device)
                    depths = batch["depth"].to(device)
                    ids = batch["id"]

                    # TTA
                    logits = model(images, depths)
                    probs = torch.sigmoid(logits)

                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = model(images_flip, depths)
                    probs_flip = torch.sigmoid(logits_flip)
                    probs_flip = torch.flip(probs_flip, dims=[3])

                    probs_avg = (probs + probs_flip) / 2.0
                    probs_np = probs_avg.cpu().numpy()

                    for i, img_id in enumerate(ids):
                        p = probs_np[i].squeeze()
                        p_unpad = unpad_image(
                            p, (Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                        )

                        if img_id not in test_probs:
                            test_probs[img_id] = np.zeros_like(p_unpad)
                        test_probs[img_id] += p_unpad

            del model
            torch.cuda.empty_cache()

        # Generate RLEs
        ids_list = []
        rles_list = []

        print(f"Encoding submission with threshold {best_thresh:.4f}...")

        # Ensure order matches test.csv
        test_df = pd.read_csv(Config.TEST_CSV)
        for img_id in test_df["id"]:
            avg_prob = test_probs[img_id] / Config.N_FOLDS
            mask_bin = (avg_prob > best_thresh).astype(np.uint8)
            rle = rle_encode(mask_bin)
            ids_list.append(img_id)
            rles_list.append(rle)

        sub_df = pd.DataFrame({"id": ids_list, "rle_mask": rles_list})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_val_map:.4f} did not meet threshold 0.7985. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
