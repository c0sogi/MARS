import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, rle_encode, calculate_iou_map
from library.dataset import make_loader, get_stratified_folds
from library.training import train_fold
from library.inference import InferenceRunner
from library.model import SaltUNetPlusPlus

# =========================================================================
# Configuration Overrides for Fast Baseline
# =========================================================================
# We override these to ensure the code completes within the 2-hour limit
# while still testing the full semi-supervised pipeline logic.
Config.EPOCHS = 20
Config.LOVASZ_EPOCH = 10
Config.NUM_FOLDS = 1  # Run only one fold for the baseline
Config.DEBUG = False  # Use full dataset (it's small enough)


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # =========================================================================
    # 1. Prepare Data Splits
    # =========================================================================
    print("\n[Step 1] Preparing Stratified Folds...")
    # We get the standard stratified splits but only use Fold 0
    folds = get_stratified_folds(n_splits=5)
    train_df_orig, val_df = folds[0]

    print(f"Train samples (Original): {len(train_df_orig)}")
    print(f"Validation samples: {len(val_df)}")

    # =========================================================================
    # 2. Stage 1: Supervised Training
    # =========================================================================
    print("\n[Step 2] Stage 1: Supervised Training (Fold 0)...")

    # Create Loaders
    train_loader_s1 = make_loader(
        train_df_orig, phase="train", cache_name="train_fold0_s1"
    )
    val_loader = make_loader(val_df, phase="val", cache_name="val_fold0")

    # Train
    best_score_s1 = train_fold(train_loader_s1, val_loader, fold_idx=0)
    print(f"Stage 1 Best mAP: {best_score_s1:.4f}")

    # =========================================================================
    # 3. Pseudo-Label Generation
    # =========================================================================
    print("\n[Step 3] Generating Pseudo-Labels for Test Set...")

    # Load Stage 1 Model
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "fold_0_best.pth")
    model_s1 = SaltUNetPlusPlus().to(device)
    model_s1.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_s1.eval()

    # Load Test Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_loader = make_loader(
        test_df, phase="test", cache_name="test_pseudo", shuffle=False
    )

    pseudo_records = []

    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)

            # Inference (Simple forward pass, no TTA for speed in this step)
            outputs = model_s1(images)

            # Resize to original 101x101
            outputs = F.interpolate(
                outputs,
                size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                mode="bilinear",
                align_corners=True,
            )
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze(1)

            # Process batch
            for i, img_id in enumerate(ids):
                # Binarize with 0.5 threshold
                mask = (probs[i] > 0.5).astype(np.uint8)
                rle = rle_encode(mask)

                # Retrieve metadata
                meta_row = test_df[test_df["id"] == img_id].iloc[0]

                pseudo_records.append(
                    {
                        "id": img_id,
                        "rle_mask": rle,
                        "z": meta_row["z"],
                        "image_path": meta_row["image_path"],
                        "coverage_class": 0,  # Dummy value, not used for training logic
                        "mask_path": None,  # Not used by loader when rle_mask is present
                    }
                )

    pseudo_df = pd.DataFrame(pseudo_records)
    print(f"Generated {len(pseudo_df)} pseudo-labels.")

    # Clean up Stage 1 model
    del model_s1
    torch.cuda.empty_cache()

    # =========================================================================
    # 4. Stage 2: Semi-Supervised Training
    # =========================================================================
    print("\n[Step 4] Stage 2: Semi-Supervised Training (Labeled + Pseudo)...")

    # Combine datasets
    combined_train_df = pd.concat([train_df_orig, pseudo_df], ignore_index=True)
    print(f"Combined Train samples: {len(combined_train_df)}")

    # Create Combined Loader
    # Note: We use a fresh cache name to ensure masks are decoded from the new RLEs
    combined_loader = make_loader(
        combined_train_df, phase="train", cache_name="train_fold0_s2"
    )

    # Train (This will overwrite the fold_0_best.pth with the Stage 2 model)
    best_score_s2 = train_fold(combined_loader, val_loader, fold_idx=0)

    # Print Final Metric exactly as requested
    print(f"Final Validation Metric: {best_score_s2}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("\n[Step 5] Failure Analysis...")

    # Load best Stage 2 model
    model_s2 = SaltUNetPlusPlus().to(device)
    model_s2.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_s2.eval()

    val_ious = []
    val_depths = []
    val_coverages = []

    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device)

            # Predict
            outputs = model_s2(images)
            outputs = F.interpolate(
                outputs,
                size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                mode="bilinear",
                align_corners=True,
            )
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy().squeeze(1)

            # Ground Truth (resize if needed)
            if masks.shape[-2:] != (Config.ORIG_HEIGHT, Config.ORIG_WIDTH):
                masks = F.interpolate(
                    masks.float(),
                    size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                    mode="nearest",
                )
            gts = masks.cpu().numpy().squeeze(1)

            # Calculate per-image IoU
            for i in range(len(ids)):
                intersection = np.sum(preds[i] * gts[i])
                union = np.sum(preds[i]) + np.sum(gts[i]) - intersection
                iou = 1.0 if union == 0 else intersection / union

                val_ious.append(iou)

                # Get metadata
                row = val_df[val_df["id"] == ids[i]].iloc[0]
                val_depths.append(row["z"])
                val_coverages.append(row["coverage"])

    # Correlations
    corr_z, _ = pearsonr(val_ious, val_depths)
    corr_cov, _ = pearsonr(val_ious, val_coverages)

    print(f"Correlation (IoU vs Depth): {corr_z:.4f}")
    print(f"Correlation (IoU vs Salt Coverage): {corr_cov:.4f}")

    # =========================================================================
    # 6. Submission
    # =========================================================================
    print("\n[Step 6] Generating Submission...")

    if best_score_s2 > 0.827:
        print("Validation metric condition met (> 0.827). Proceeding with submission.")

        # Use InferenceRunner to generate submission
        # Note: Since Config.NUM_FOLDS=1, it will only look for fold_0_best.pth
        runner = InferenceRunner(device=device)
        runner.generate_submission(threshold=0.5)

    else:
        print(f"Validation metric {best_score_s2} <= 0.827. Submission skipped.")


if __name__ == "__main__":
    main()
