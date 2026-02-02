import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.trainer import SaltTrainer
from library.utils import seed_everything, rle_encode, calculate_iou_map
from library.dataset import SaltDataset, get_transforms
from library.model import SaltUNetPlusPlus


def main():
    # 1. Setup and Configuration Override for Fast Baseline
    seed_everything(Config.SEED)

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("=== Starting Runfile Execution ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Training
    trainer = SaltTrainer()
    trainer.fit()

    # 3. Post-Training Validation & Threshold Optimization
    print("\n=== Performing Final Validation & Threshold Optimization ===")

    # Load Best Model
    device = torch.device(Config.DEVICE)
    model = SaltUNetPlusPlus().to(device)
    checkpoint_path = Config.BEST_MODEL_PATH

    if not os.path.exists(checkpoint_path):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Setup Validation Loader
    val_dataset = SaltDataset(
        mode="val", transform=get_transforms(mode="val"), load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_masks = []
    all_ids = []

    # Validation Inference with TTA
    with torch.no_grad():
        for images, masks, ids in tqdm(val_loader, desc="Validating", disable=True):
            images = images.to(device)

            # TTA: Original
            out = model(images)
            pred = torch.sigmoid(out)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, [3])
            out_flip = model(images_flip)
            pred_flip = torch.flip(torch.sigmoid(out_flip), [3])

            # Average
            avg_pred = (pred + pred_flip) / 2.0

            all_preds.append(avg_pred.cpu().numpy())
            all_masks.append(masks.numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Squeeze channels: (B, 1, H, W) -> (B, H, W)
    if all_preds.ndim == 4:
        all_preds = all_preds.squeeze(1)
    if all_masks.ndim == 4:
        all_masks = all_masks.squeeze(1)

    # Threshold Optimization
    thresholds = np.linspace(0.2, 0.8, 61)  # Wider fine-grained sweep
    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        score = calculate_iou_map(all_preds, all_masks, pixel_threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"Final Validation Metric: {best_score}")
    print(f"Optimal Threshold: {best_threshold:.4f}")

    # 4. Failure Analysis
    print("\n=== Performing Failure Analysis ===")

    # Calculate per-sample IoU at best threshold
    preds_bin = (all_preds > best_threshold).astype(np.uint8)
    masks_bin = (all_masks > 0.5).astype(np.uint8)

    ious = []
    for i in range(len(preds_bin)):
        p = preds_bin[i].flatten()
        t = masks_bin[i].flatten()

        sum_p = np.sum(p)
        sum_t = np.sum(t)

        if sum_p == 0 and sum_t == 0:
            iou = 1.0
        elif sum_p > 0 and sum_t == 0:
            iou = 0.0
        elif sum_p == 0 and sum_t > 0:
            iou = 0.0
        else:
            intersection = np.sum((p * t) > 0)
            union = np.sum((p + t) > 0)
            iou = intersection / union
        ious.append(iou)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {"id": all_ids, "iou": ious, "error": 1.0 - np.array(ious)}
    )

    # Load Metadata to get features
    val_meta = pd.read_csv(Config.VAL_METADATA)
    analysis_df = analysis_df.merge(
        val_meta[["id", "z", "coverage"]], on="id", how="left"
    )

    # Calculate Correlations
    corr_depth = analysis_df["error"].corr(analysis_df["z"])
    corr_cov = analysis_df["error"].corr(analysis_df["coverage"])

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 5. Submission
    if best_score > 0.827:
        print("\n=== Generating Submission ===")

        test_dataset = SaltDataset(
            mode="test", transform=get_transforms(mode="test"), load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_data = []

        with torch.no_grad():
            for images, _, ids in tqdm(test_loader, desc="Inference", disable=True):
                images = images.to(device)

                # TTA
                out = model(images)
                pred = torch.sigmoid(out)

                images_flip = torch.flip(images, [3])
                out_flip = model(images_flip)
                pred_flip = torch.flip(torch.sigmoid(out_flip), [3])

                avg_pred = (pred + pred_flip) / 2.0

                # Binarize
                pred_bin = (avg_pred > best_threshold).byte().cpu().numpy()

                # Squeeze
                if pred_bin.ndim == 4:
                    pred_bin = pred_bin.squeeze(1)

                # Encode
                for i, img_id in enumerate(ids):
                    mask = pred_bin[i]
                    # Resize back to original 101x101 if needed (Dataset pads to 128x128)
                    # The dataset uses reflection padding. We need to crop to 101x101.
                    # Padding was: PadIfNeeded(min_height=128, min_width=128, border_mode=reflect)
                    # Albumentations PadIfNeeded centers the image if possible or pads.
                    # Since 101 -> 128, it likely pads 13 on one side and 14 on other or similar.
                    # However, Albumentations `PadIfNeeded` behavior with `position` defaults to center.
                    # Let's verify crop.
                    # 128 - 101 = 27. 13 top/left, 14 bottom/right usually.
                    # To be safe, we can just center crop 101x101.

                    h, w = mask.shape
                    diff_h = (h - 101) // 2
                    diff_w = (w - 101) // 2

                    mask_cropped = mask[diff_h : diff_h + 101, diff_w : diff_w + 101]

                    rle = rle_encode(mask_cropped)
                    submission_data.append([img_id, rle])

        sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {best_score:.4f} did not meet threshold 0.827. Skipping submission."
        )


if __name__ == "__main__":
    main()
