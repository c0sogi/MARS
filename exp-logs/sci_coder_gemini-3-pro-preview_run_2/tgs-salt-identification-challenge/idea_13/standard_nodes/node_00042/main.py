import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import library modules
from library.config import Config

# =============================================================================
# Configuration Overrides for Fast Baseline & Submission
# =============================================================================
# Reduce epochs to ensure execution finishes within 2 hours
Config.EPOCHS = 20
Config.T_MAX = 20
# Set submission path as per instructions
Config.SUBMISSION_PATH = "./submission/submission.csv"
# Ensure submission directory exists
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

# Import remaining library functions after Config setup
from library.dataset import get_dataloaders
from library.model import SaltNet
from library.train import train_model
from library.utils import do_kaggle_metric, rle_encode, unpad_image


def main():
    print("Initializing Salt Segmentation Pipeline...")

    # =========================================================================
    # 1. Training
    # =========================================================================
    print(f"Starting training for {Config.EPOCHS} epochs...")
    # train_model handles the loop, validation, and saves best_model.pth to Config.BEST_MODEL_PATH
    train_model(load_cached_data=True)

    # =========================================================================
    # 2. Load Best Model
    # =========================================================================
    device = torch.device(Config.DEVICE)
    model = SaltNet().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        return

    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # =========================================================================
    # 3. Validation & Threshold Optimization
    # =========================================================================
    print("Loading dataloaders...")
    _, val_loader, test_loader = get_dataloaders(use_cache=True)

    print("Running validation inference with TTA...")
    val_preds = []
    val_masks = []
    val_ids = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(device)
            depths = depths.to(device)  # Note: dataset sets z=0.0 for validation

            # TTA: Original
            out1 = torch.sigmoid(model(images, depths))

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, [3])
            out2 = torch.sigmoid(model(images_flip, depths))
            out2 = torch.flip(out2, [3])  # Flip back

            # Average
            pred = (out1 + out2) / 2.0

            val_preds.append(pred.cpu().numpy())
            val_masks.append(masks.numpy())
            val_ids.extend(ids)

    val_preds = np.concatenate(val_preds)  # (N, 1, 128, 128)
    val_masks = np.concatenate(val_masks)  # (N, 1, 128, 128)

    print("Optimizing binarization threshold...")
    thresholds = np.linspace(0.3, 0.7, 21)  # Sweep from 0.3 to 0.7
    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        # do_kaggle_metric calculates mAP over IoU thresholds (0.5-0.95)
        # We pass 't' as the pixel binarization threshold
        score = do_kaggle_metric(val_preds, val_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"Best Pixel Threshold: {best_threshold:.4f}")
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {best_score}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n--- Failure Analysis ---")

    # Load metadata to get TRUE depths (since loader gave us 0.0)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    id_to_depth = dict(zip(val_meta["id"], val_meta["z"]))

    errors = []
    true_depths = []
    salt_coverages = []

    # Binarize predictions with best threshold
    preds_bin = (
        (val_preds > best_threshold).astype(np.uint8).squeeze(1)
    )  # (N, 128, 128)
    masks_bin = (val_masks > 0.5).astype(np.uint8).squeeze(1)  # (N, 128, 128)

    for i, img_id in enumerate(val_ids):
        # Unpad to original size for accurate analysis
        p_img = unpad_image(preds_bin[i])
        m_img = unpad_image(masks_bin[i])

        # Calculate metric for single image
        # do_kaggle_metric expects (Batch, H, W)
        s = do_kaggle_metric(p_img[None, ...], m_img[None, ...], threshold=0.5)
        errors.append(1.0 - s)

        # Get True Depth
        true_depths.append(id_to_depth.get(img_id, 0))

        # Calculate Salt Coverage
        salt_coverages.append(np.mean(m_img))

    errors = np.array(errors)
    true_depths = np.array(true_depths)
    salt_coverages = np.array(salt_coverages)

    if len(errors) > 1:
        corr_depth, _ = pearsonr(errors, true_depths)
        corr_cov, _ = pearsonr(errors, salt_coverages)

        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")
    else:
        print("Insufficient samples for correlation analysis.")

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    TARGET_SCORE = 0.7985

    if best_score > TARGET_SCORE:
        print(
            f"\nValidation score {best_score:.4f} > {TARGET_SCORE}. Generating submission..."
        )

        submission_data = []

        with torch.no_grad():
            for images, depths, ids in test_loader:
                images = images.to(device)
                depths = depths.to(device)  # Loader sets z=0.0 for test

                # TTA: Original
                out1 = torch.sigmoid(model(images, depths))

                # TTA: Horizontal Flip
                images_flip = torch.flip(images, [3])
                out2 = torch.sigmoid(model(images_flip, depths))
                out2 = torch.flip(out2, [3])

                # Average
                pred_batch = (out1 + out2) / 2.0
                pred_batch = pred_batch.cpu().numpy()  # (B, 1, 128, 128)

                for i in range(len(ids)):
                    img_id = ids[i]

                    # Binarize
                    mask_pred = (pred_batch[i, 0] > best_threshold).astype(np.uint8)

                    # Unpad (128x128 -> 101x101)
                    mask_pred = unpad_image(mask_pred)

                    # RLE Encode
                    rle = rle_encode(mask_pred)
                    submission_data.append([img_id, rle])

        sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score {best_score:.4f} <= {TARGET_SCORE}. Submission skipped."
        )


if __name__ == "__main__":
    main()
