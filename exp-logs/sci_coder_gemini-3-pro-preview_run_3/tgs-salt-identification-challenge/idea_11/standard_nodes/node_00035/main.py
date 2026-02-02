import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Import library modules
from library.utils import seed_everything, rle_encode, calculate_map_score
from library.dataset import get_dataloaders, get_test_loader, load_or_create_data
from library.model import SaltUNetPlusPlus
from library.loss import BCEDiceLoss, LovaszHingeLoss
from library.engine import train_one_epoch, validate_one_epoch

# --- Configuration ---
SEED = 42
NUM_FOLDS = 5
EPOCHS = 30  # 15 Warmup + 15 Finetune
SWITCH_EPOCH = 15
BATCH_SIZE = 64
NUM_WORKERS = 2
LR_WARMUP = 1e-3
LR_FINETUNE = 5e-5  # Conservative LR for Lovasz
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBMISSION_THRESHOLD = 0.827
OUTPUT_DIR = "./working"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submission")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def main():
    seed_everything(SEED)
    print(f"Starting execution on {DEVICE}")

    # --- Data Loading (Cache Creation) ---
    # Trigger cache creation once before loops
    print("Initializing data cache...")
    load_or_create_data(mode="train", load_cached_data=True)
    load_or_create_data(mode="test", load_cached_data=True)

    # --- Storage for OOF ---
    oof_preds_dict = {}  # id -> pred_mask (prob)
    oof_targets_dict = {}  # id -> target_mask
    oof_depths_dict = {}  # id -> depth
    oof_ids_list = []

    fold_scores = []

    # --- Cross-Validation Loop ---
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*20} Fold {fold+1}/{NUM_FOLDS} {'='*20}")

        # 1. Data Loaders
        train_loader, val_loader = get_dataloaders(
            fold_idx=fold,
            n_folds=NUM_FOLDS,
            batch_size=BATCH_SIZE,
            load_cached_data=True,
            num_workers=NUM_WORKERS,
        )

        # 2. Model Initialization
        model = SaltUNetPlusPlus(
            encoder_name="resnext50_32x4d",
            in_channels=3,
            classes=1,
            deep_supervision=True,
        ).to(DEVICE)

        # 3. Optimizer & Scaler
        optimizer = optim.AdamW(model.parameters(), lr=LR_WARMUP, weight_decay=1e-4)
        scaler = GradScaler()

        # 4. Schedulers
        # We use ReduceLROnPlateau, but we also manually intervene at SWITCH_EPOCH
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=False
        )

        # 5. Loss Functions
        criterion_bce = BCEDiceLoss().to(DEVICE)
        criterion_lovasz = LovaszHingeLoss().to(DEVICE)
        current_criterion = criterion_bce

        best_map = 0.0
        best_model_path = os.path.join(CHECKPOINT_DIR, f"fold_{fold}_best.pth")

        # --- Training Loop ---
        for epoch in range(1, EPOCHS + 1):
            # Curriculum Switch
            if epoch == SWITCH_EPOCH + 1:
                print(
                    f"   [Curriculum] Switching to Lovasz-Hinge Loss & Resetting LR to {LR_FINETUNE}"
                )
                current_criterion = criterion_lovasz
                # Re-initialize optimizer or set param groups
                for param_group in optimizer.param_groups:
                    param_group["lr"] = LR_FINETUNE

            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, current_criterion, DEVICE
            )

            # Validate
            val_loss, val_map = validate_one_epoch(
                model, val_loader, current_criterion, DEVICE
            )

            # Scheduler Step
            scheduler.step(val_map)

            # Save Best
            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), best_model_path)

            # Logging (Minimal)
            # print(f"   Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}")

        print(f"   Best mAP for Fold {fold}: {best_map:.4f}")
        fold_scores.append(best_map)

        # --- Generate OOF Predictions for this Fold ---
        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model.eval()
        model.deep_supervision = False  # Disable for inference

        # Cropping indices (128 -> 101)
        start_idx = (128 - 101) // 2
        end_idx = start_idx + 101

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)  # (B, 1, 128, 128)
                ids = batch["id"]

                # Forward (TTA)
                logits = model(images)
                probs = torch.sigmoid(logits)

                logits_flip = model(torch.flip(images, dims=[3]))
                probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])

                avg_probs = (probs + probs_flip) / 2.0

                # Crop
                preds_cropped = (
                    avg_probs[:, :, start_idx:end_idx, start_idx:end_idx].cpu().numpy()
                )
                masks_cropped = (
                    masks[:, :, start_idx:end_idx, start_idx:end_idx].cpu().numpy()
                )

                # Store
                for i, img_id in enumerate(ids):
                    oof_preds_dict[img_id] = preds_cropped[i, 0]
                    oof_targets_dict[img_id] = masks_cropped[i, 0]
                    oof_ids_list.append(img_id)

    # --- Global Threshold Optimization ---
    print("\n" + "=" * 20 + " Global Optimization " + "=" * 20)

    # Convert OOF dicts to arrays aligned by ID
    sorted_ids = sorted(oof_ids_list)
    all_preds = np.array([oof_preds_dict[i] for i in sorted_ids])
    all_targets = np.array([oof_targets_dict[i] for i in sorted_ids])

    # Flatten for metric calculation efficiency
    # But calculate_map_score expects (B, H, W)

    thresholds = np.arange(0.3, 0.8, 0.05)
    best_global_threshold = 0.5
    best_global_score = 0.0

    # We can use a subset or batching if memory is tight, but 3000x101x101 is small (~30MB)
    # Perform sweep
    for t in thresholds:
        score = calculate_map_score(all_preds, all_targets, decision_threshold=t)
        if score > best_global_score:
            best_global_score = score
            best_global_threshold = t

    print(f"Optimal Threshold: {best_global_threshold:.2f}")
    print(f"Final Validation Metric: {best_global_score:.10f}")

    # --- Failure Analysis ---
    print("\n" + "=" * 20 + " Failure Analysis " + "=" * 20)

    # 1. Calculate per-image mAP at optimal threshold
    # Re-implementing simplified per-image logic here for analysis
    ious = []
    # Thresholds for the metric (0.5 to 0.95)
    metric_thresholds = np.arange(0.5, 0.96, 0.05)

    # Binarize with optimal threshold
    binary_preds = (all_preds > best_global_threshold).astype(np.uint8)
    binary_targets = (all_targets > 0.5).astype(np.uint8)

    per_image_scores = []

    for i in range(len(all_preds)):
        p = binary_preds[i].flatten()
        t = binary_targets[i].flatten()

        intersection = (p & t).sum()
        union = (p | t).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate AP for this image
        # AP = mean of (iou > metric_t)
        matches = iou > metric_thresholds
        ap = matches.mean()
        per_image_scores.append(ap)

    per_image_scores = np.array(per_image_scores)

    # 2. Get Metadata for Correlation
    # We need depth and coverage for these IDs.
    # Load metadata
    df_train = pd.read_csv("./metadata/train_metadata.csv")
    df_val = pd.read_csv("./metadata/val_metadata.csv")
    df_meta = pd.concat([df_train, df_val])
    df_meta = df_meta.set_index("id")

    depths = []
    coverages = []

    for img_id in sorted_ids:
        row = df_meta.loc[img_id]
        depths.append(row["z"])
        coverages.append(row["coverage"])

    depths = np.array(depths)
    coverages = np.array(coverages)

    # 3. Calculate Correlations
    # We correlate Error (1 - Score) with features
    errors = 1.0 - per_image_scores

    corr_depth, _ = pearsonr(errors, depths)
    corr_cov, _ = pearsonr(errors, coverages)

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # --- Submission ---
    if best_global_score > SUBMISSION_THRESHOLD:
        print("\n" + "=" * 20 + " Generating Submission " + "=" * 20)

        test_loader = get_test_loader(
            batch_size=BATCH_SIZE, load_cached_data=True, num_workers=NUM_WORKERS
        )

        # Load all 5 models
        models = []
        for fold in range(NUM_FOLDS):
            m = SaltUNetPlusPlus(
                encoder_name="resnext50_32x4d",
                in_channels=3,
                classes=1,
                deep_supervision=False,
            ).to(DEVICE)
            m.load_state_dict(
                torch.load(
                    os.path.join(CHECKPOINT_DIR, f"fold_{fold}_best.pth"),
                    map_location=DEVICE,
                )
            )
            m.eval()
            models.append(m)

        submission_data = []

        start_idx = (128 - 101) // 2
        end_idx = start_idx + 101

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(DEVICE)
                ids = batch["id"]

                # Ensemble Prediction
                batch_preds = 0.0

                # TTA Flip
                images_flip = torch.flip(images, dims=[3])

                for m in models:
                    # Normal
                    logits = m(images)
                    probs = torch.sigmoid(logits)

                    # Flip
                    logits_flip = m(images_flip)
                    probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])

                    batch_preds += (probs + probs_flip) / 2.0

                batch_preds /= len(models)

                # Crop
                preds_cropped = (
                    batch_preds[:, :, start_idx:end_idx, start_idx:end_idx]
                    .cpu()
                    .numpy()
                )

                # Threshold and Encode
                for i, img_id in enumerate(ids):
                    pred_mask = (preds_cropped[i, 0] > best_global_threshold).astype(
                        np.uint8
                    )
                    rle = rle_encode(pred_mask)
                    submission_data.append({"id": img_id, "rle_mask": rle})

        sub_df = pd.DataFrame(submission_data)
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nScore {best_global_score:.4f} did not meet threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
