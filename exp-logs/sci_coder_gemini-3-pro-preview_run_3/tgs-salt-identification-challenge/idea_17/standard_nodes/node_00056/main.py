import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    rle_encode,
    calc_map,
    iou_metric,
)
from library.dataset import get_loaders, get_test_loader
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss, DeepSupervisionLoss
from library.engine import train_one_epoch, evaluate


def main():
    # 1. Setup and Configuration Overrides for Fast Baseline
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for fast execution within time limits
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100  # Small subset for speed
    Config.TOTAL_EPOCHS = 2  # Minimal epochs to demonstrate pipeline
    Config.NUM_FOLDS = 1  # Single fold for baseline

    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Starting Fast Baseline Run on {device}...")

    # 2. Data Loading (Fold 0)
    train_loader, val_loader = get_loaders(
        fold=0, debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Model Initialization
    model = SaltUNetPlusPlus()
    model.to(device)

    # 4. Optimization Setup
    # Phase 1 settings initially
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR_PHASE1, weight_decay=1e-4)
    scaler = GradScaler()

    # Losses
    bce_dice_loss = BCEDiceLoss()
    lovasz_loss = LovaszHingeLoss()

    # Deep Supervision Wrappers
    # Weights for 4 heads: [0.1, 0.1, 0.1, 1.0] roughly, or equal as per idea
    ds_weights = [0.5, 0.5, 0.5, 1.0]
    criterion_phase1 = DeepSupervisionLoss(bce_dice_loss, weights=ds_weights)
    criterion_phase2 = DeepSupervisionLoss(
        lovasz_loss, weights=None
    )  # Only final head used in logic

    best_map = 0.0
    current_phase = 1

    # 5. Training Loop
    for epoch in range(Config.TOTAL_EPOCHS):
        start_time = time.time()

        # Adaptive Phase Switching Logic (Simplified for 2-epoch baseline)
        # Epoch 0: Phase 1, Epoch 1: Phase 2
        if epoch == 1:
            current_phase = 2
            # Update LR for Phase 2
            for param_group in optimizer.param_groups:
                param_group["lr"] = Config.LR_PHASE2

        # Select Loss and Deep Supervision Mode
        if current_phase == 1:
            loss_fn = criterion_phase1
            deep_supervision = True
        else:
            loss_fn = criterion_phase2  # Will effectively use Lovasz on final head
            deep_supervision = False

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, deep_supervision
        )

        # Validate
        val_loss, val_map = evaluate(
            model, val_loader, lovasz_loss, device, deep_supervision=False
        )

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.TOTAL_EPOCHS} [Phase {current_phase}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f} | Time: {elapsed:.1f}s"
        )

        # Save Best
        if val_map > best_map:
            best_map = val_map
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                is_best=True,
                checkpoint_dir=Config.CHECKPOINT_DIR,
                filename="fold0_best.pth",
            )

    # 6. Final Validation & Failure Analysis
    print("\n--- Final Validation & Failure Analysis ---")

    # Load best model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device)

    model.eval()

    # Re-run validation inference to get per-image metrics for failure analysis
    # We need to manually iterate to link predictions back to metadata (depth)
    val_ious = []
    val_depths = []
    val_ids = []

    # Access underlying dataset to get depths
    val_dataset = val_loader.dataset
    # Create a map of id -> depth
    id_to_depth = {
        id_: depth for id_, depth in zip(val_dataset.ids, val_dataset.depths)
    }

    # Crop indices for metric calculation
    start_idx = (Config.MODEL_HEIGHT - Config.ORIG_HEIGHT) // 2
    end_idx = start_idx + Config.ORIG_HEIGHT

    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]

            probs = torch.sigmoid(outputs)

            # Crop
            probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]
            masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]

            # Calculate IoU per image
            probs_np = probs_cropped.cpu().numpy()
            masks_np = masks_cropped.cpu().numpy()

            for i in range(len(ids)):
                # Binarize at 0.5 for IoU analysis
                p = (probs_np[i, 0] > 0.5).astype(np.uint8)
                t = (masks_np[i, 0] > 0.5).astype(np.uint8)
                iou = iou_metric(p, t)

                val_ious.append(iou)
                val_ids.append(ids[i])
                val_depths.append(id_to_depth[ids[i]])

    # Calculate final metric on the whole validation set
    final_metric = np.mean(
        [calc_map(np.array([p]), np.array([t])) for p, t in zip(probs_np, masks_np)]
    )
    # Note: The above line is an approximation using the last batch.
    # Correct way is to use the best_map computed during the loop or re-compute on all.
    # Since we saved best_map, let's use that as the authoritative metric for the run.
    print(f"Final Validation Metric: {best_map:.10f}")

    # Failure Analysis: Correlation
    errors = 1.0 - np.array(val_ious)
    depths = np.array(val_depths)

    if len(errors) > 1:
        corr_depth, _ = pearsonr(errors, depths)
        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 7. Submission Generation
    # Condition: Metric > 0.827
    if best_map > 0.827:
        print("\nMetric threshold met. Generating submission...")

        test_loader = get_test_loader(load_cached_data=True)
        submission_data = []

        model.eval()
        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                # TTA: Forward
                out = model(images)
                if isinstance(out, (list, tuple)):
                    out = out[-1]
                probs = torch.sigmoid(out)

                # TTA: Horizontal Flip
                if Config.USE_TTA:
                    images_flip = torch.flip(images, dims=[3])
                    out_flip = model(images_flip)
                    if isinstance(out_flip, (list, tuple)):
                        out_flip = out_flip[-1]
                    probs_flip = torch.sigmoid(out_flip)
                    probs_flip = torch.flip(probs_flip, dims=[3])
                    probs = (probs + probs_flip) / 2.0

                # Crop to 101x101
                probs = probs[:, :, start_idx:end_idx, start_idx:end_idx]
                probs_np = probs.cpu().numpy()

                # Encode
                for i in range(len(ids)):
                    # Threshold at 0.5 (or optimized threshold)
                    mask_bin = (probs_np[i, 0] > 0.5).astype(np.uint8)
                    rle = rle_encode(mask_bin)
                    submission_data.append([ids[i], rle])

        # Save
        sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {best_map:.4f} <= 0.827. Skipping submission generation.")


if __name__ == "__main__":
    main()
