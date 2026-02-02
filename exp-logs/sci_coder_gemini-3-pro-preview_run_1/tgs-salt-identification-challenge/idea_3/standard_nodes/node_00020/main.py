import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import cv2
import gc
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.utils import set_seed, rle_encode, calculate_map_score, calculate_iou
from library.model import DepthAwareUNet
from library.dataset import SaltDataset, get_transforms
from library.train import run_fold


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    # Increased epochs to 60 for full convergence (Cite Lesson 11)
    EPOCHS = 60
    SEED = 42

    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Running on {DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Single Model Training
    # -------------------------------------------------------------------------
    # Train on full training set (Cite Lesson 19)
    print(f"\n--- Starting Single Model Training ---")

    model_dir = os.path.join(WORKING_DIR, "single_model")
    os.makedirs(model_dir, exist_ok=True)

    run_fold(
        train_metadata=TRAIN_META,
        val_metadata=VAL_META,
        output_dir=model_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        device=DEVICE,
        num_workers=2,
        base_filters=32,
    )

    model_path = os.path.join(model_dir, "best_model.pth")

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. Hold-out Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Performing Hold-out Validation ---")

    # Load hold-out set
    val_dataset = SaltDataset(
        metadata_csv=VAL_META,
        transform=get_transforms(mode="val"),  # No augmentation for inference
        mode="val",
        cache_dir=WORKING_DIR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load the single trained model
    models = []
    m = DepthAwareUNet(input_channels=2, num_classes=1, base_filters=32)
    m.load_state_dict(torch.load(model_path, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    models.append(m)

    # Inference Loop
    all_preds = []
    all_truths = []
    all_ids = []

    # Crop constants (128 -> 101)
    pad_top, pad_left = 13, 13
    orig_h, orig_w = 101, 101

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            ids = batch["id"]

            batch_preds = torch.zeros_like(masks)

            # Ensemble + TTA (Horizontal Flip)
            for model in models:
                # Original
                out = torch.sigmoid(model(images))
                batch_preds += out

                # Flip
                images_flip = torch.flip(images, [3])
                out_flip = torch.sigmoid(model(images_flip))
                batch_preds += torch.flip(out_flip, [3])

            # Average (5 models * 2 TTA = 10 predictions)
            batch_preds /= len(models) * 2

            # Move to CPU and Crop back to original size
            probs = batch_preds.cpu().numpy()
            true_masks = masks.cpu().numpy()

            for i in range(probs.shape[0]):
                # Crop center
                p_crop = probs[
                    i, 0, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
                ]
                t_crop = true_masks[
                    i, 0, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
                ]

                all_preds.append(p_crop)
                all_truths.append(t_crop)
                all_ids.append(ids[i])

    # Calculate Metric
    # Threshold at 0.5 for binary mask generation for mAP calculation
    bin_preds = [p > 0.5 for p in all_preds]
    bin_truths = [t > 0.5 for t in all_truths]

    final_metric = calculate_map_score(bin_preds, bin_truths)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate IoU per image to correlate with metadata
    ious = []
    for p, t in zip(bin_preds, bin_truths):
        ious.append(calculate_iou(p, t))

    df_val = pd.read_csv(VAL_META)
    # Map IoUs to IDs
    id_to_iou = dict(zip(all_ids, ious))
    df_val["iou"] = df_val["id"].map(id_to_iou)

    # Correlation Analysis
    corr_depth, _ = pearsonr(df_val["z"], df_val["iou"])
    corr_cov, _ = pearsonr(df_val["coverage"], df_val["iou"])

    print(f"Correlation (Depth vs IoU): {corr_depth:.4f}")
    print(f"Correlation (Salt Coverage vs IoU): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.7647:
        print("\n--- Generating Submission ---")

        test_dataset = SaltDataset(
            metadata_csv=TEST_META,
            transform=get_transforms(mode="test"),
            mode="test",
            cache_dir=WORKING_DIR,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(DEVICE)
                ids = batch["id"]

                # Setup accumulator for ensemble predictions
                # Output shape is (B, 1, 128, 128) due to padding
                batch_preds = torch.zeros((images.size(0), 1, 128, 128), device=DEVICE)

                for model in models:
                    # Original
                    out = torch.sigmoid(model(images))
                    batch_preds += out

                    # Flip
                    images_flip = torch.flip(images, [3])
                    out_flip = torch.sigmoid(model(images_flip))
                    batch_preds += torch.flip(out_flip, [3])

                batch_preds /= len(models) * 2

                probs = batch_preds.cpu().numpy()

                for i in range(probs.shape[0]):
                    # Crop
                    p_crop = probs[
                        i, 0, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
                    ]

                    # Threshold
                    mask_bin = (p_crop > 0.5).astype(np.uint8)

                    # RLE Encode
                    rle = rle_encode(mask_bin)
                    submission_rows.append([ids[i], rle])

        # Save Submission
        df_sub = pd.DataFrame(submission_rows, columns=["id", "rle_mask"])
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(f"Metric {final_metric} <= 0.7647. Skipping submission.")


if __name__ == "__main__":
    main()
