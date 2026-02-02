import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, calc_map, rle_encode
from library.dataset import SaltDataset, get_transforms, load_data
from library.model import SaltModel
from library.train import run_fold


def optimize_threshold(preds, targets):
    """
    Sweeps through probability thresholds to maximize mAP.
    """
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        score = calc_map(preds, targets, pixel_threshold=t)
        if score > best_score:
            best_score = score
            best_t = t

    return best_score, best_t


def analyze_failures(preds, targets, ids, best_threshold):
    """
    Correlates error with depth and coverage.
    """
    # Calculate per-image score (1 - mAP)
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    p_bin = (preds > best_threshold).astype(np.uint8)
    t_bin = (targets > 0.5).astype(np.uint8)

    # Flatten spatial dims
    p_flat = p_bin.reshape(p_bin.shape[0], -1)
    t_flat = t_bin.reshape(t_bin.shape[0], -1)

    intersection = (p_flat & t_flat).sum(axis=1)
    union = (p_flat | t_flat).sum(axis=1)

    # Avoid div by zero
    iou = np.ones_like(intersection, dtype=np.float32)
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Compare against thresholds
    matches = iou[:, None] > iou_thresholds[None, :]
    precisions = matches.mean(axis=1)
    errors = 1.0 - precisions

    # Load metadata to match IDs
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    id_to_depth = dict(zip(df["id"], df["z"]))
    id_to_cov = dict(zip(df["id"], df["coverage"]))

    depths = np.array([id_to_depth[i] for i in ids])
    coverages = np.array([id_to_cov[i] for i in ids])

    # Correlations
    corr_depth, _ = pearsonr(errors, depths)
    corr_cov, _ = pearsonr(errors, coverages)

    print(f"Failure Analysis - Correlation with Depth: {corr_depth:.4f}")
    print(f"Failure Analysis - Correlation with Salt Coverage: {corr_cov:.4f}")


def main():
    # 1. Configuration Override for Fast Baseline
    # Reducing epochs to ensure completion within 2 hours while maintaining convergence
    Config.EPOCHS = 40
    Config.setup()
    seed_everything(Config.SEED)

    print("Starting Stratified 5-Fold Ensemble Pipeline...")

    # 2. Train Folds
    for fold_idx in range(Config.N_FOLDS):
        run_fold(fold_idx)

    # 3. OOF Inference & Aggregation
    print("\nGenerating OOF Predictions...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    ids, images, masks, depths = load_data(df_train, "train", load_cached_data=True)

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    all_oof_preds = []
    all_oof_targets = []
    all_oof_ids = []

    device = Config.DEVICE

    # Calculate crop indices to revert padding (128 -> 101)
    pad_h = Config.IMG_SIZE - Config.ORIG_SIZE
    pad_w = Config.IMG_SIZE - Config.ORIG_SIZE
    h_start = pad_h // 2
    w_start = pad_w // 2

    for fold_idx, (_, val_idx) in enumerate(skf.split(ids, df_train["coverage_class"])):
        print(f"Processing Fold {fold_idx} Validation...")

        # Load Model
        model = SaltModel(
            encoder_name=Config.ENCODER, pretrained=False, in_channels=Config.CHANNELS
        )
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        # Data
        val_dataset = SaltDataset(
            images[val_idx],
            depths[val_idx],
            masks[val_idx],
            transform=get_transforms("valid"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device)

                # TTA: Original
                out1 = torch.sigmoid(model(imgs))

                # TTA: Horizontal Flip
                imgs_flip = torch.flip(imgs, dims=[3])
                out2 = torch.sigmoid(model(imgs_flip))
                out2 = torch.flip(out2, dims=[3])

                # Average
                avg_pred = (out1 + out2) / 2.0

                # Crop back to 101x101
                avg_pred = avg_pred[
                    :,
                    :,
                    h_start : h_start + Config.ORIG_SIZE,
                    w_start : w_start + Config.ORIG_SIZE,
                ]

                fold_preds.append(avg_pred.cpu().numpy())
                fold_targets.append(msks.numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)  # (N, 1, 101, 101)
        fold_targets = np.concatenate(fold_targets, axis=0)  # (N, H, W) or (N, 1, H, W)

        # Crop targets (dataset padding)
        if fold_targets.ndim == 4:
            fold_targets = fold_targets[
                :,
                :,
                h_start : h_start + Config.ORIG_SIZE,
                w_start : w_start + Config.ORIG_SIZE,
            ]
            fold_targets = fold_targets.squeeze(1)
        elif fold_targets.ndim == 3:
            fold_targets = fold_targets[
                :,
                h_start : h_start + Config.ORIG_SIZE,
                w_start : w_start + Config.ORIG_SIZE,
            ]

        fold_preds = fold_preds.squeeze(1)

        all_oof_preds.append(fold_preds)
        all_oof_targets.append(fold_targets)
        all_oof_ids.extend(ids[val_idx])

        del model
        torch.cuda.empty_cache()

    all_oof_preds = np.concatenate(all_oof_preds, axis=0)
    all_oof_targets = np.concatenate(all_oof_targets, axis=0)

    # 4. Threshold Optimization
    print("Optimizing Threshold...")
    best_score, best_threshold = optimize_threshold(all_oof_preds, all_oof_targets)

    print(f"Final Validation Metric: {best_score:.10f}")
    print(f"Optimal Threshold: {best_threshold:.2f}")

    # 5. Failure Analysis
    analyze_failures(all_oof_preds, all_oof_targets, all_oof_ids, best_threshold)

    # 6. Submission
    if best_score > 0.827:
        print("Generating Submission...")

        # Load Test Data
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_ids, test_images, _, test_depths = load_data(
            df_test, "test", load_cached_data=True
        )

        test_dataset = SaltDataset(
            test_images, test_depths, masks=None, transform=get_transforms("valid")
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Load all models
        models = []
        for fold_idx in range(Config.N_FOLDS):
            m = SaltModel(
                encoder_name=Config.ENCODER,
                pretrained=False,
                in_channels=Config.CHANNELS,
            )
            m.load_state_dict(
                torch.load(
                    os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth"),
                    map_location=device,
                )
            )
            m.to(device)
            m.eval()
            models.append(m)

        final_rles = []

        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)

                # Ensemble TTA
                batch_preds = 0.0

                for m in models:
                    # Original
                    out = torch.sigmoid(m(imgs))
                    # Flip
                    imgs_flip = torch.flip(imgs, dims=[3])
                    out_flip = torch.sigmoid(m(imgs_flip))
                    out_flip = torch.flip(out_flip, dims=[3])

                    batch_preds += (out + out_flip) / 2.0

                batch_preds /= len(models)

                # Crop
                batch_preds = batch_preds[
                    :,
                    :,
                    h_start : h_start + Config.ORIG_SIZE,
                    w_start : w_start + Config.ORIG_SIZE,
                ]

                # Threshold
                batch_masks = (batch_preds > best_threshold).float()

                # RLE Encode
                batch_masks = batch_masks.cpu().numpy().squeeze(1)

                for mask in batch_masks:
                    rle = rle_encode(mask)
                    final_rles.append(rle)

        # Save
        sub_df = pd.DataFrame({"id": test_ids, "rle_mask": final_rles})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print("Validation score too low. Skipping submission.")


if __name__ == "__main__":
    main()
