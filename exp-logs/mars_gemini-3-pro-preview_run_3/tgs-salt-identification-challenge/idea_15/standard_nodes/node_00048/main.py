import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, calc_map, rle_encode, do_kaggle_metric
from library.dataset import get_dataloaders, get_test_loader
from library.model import SaltUNetPlusPlus
from library.engine import SaltEngine


def main():
    # 1. Setup and Configuration Overrides for Fast Baseline
    set_seed(Config.SEED)

    # Override epochs for a fast but effective run within 2 hours
    # 5 folds * 15 epochs * ~30s/epoch = ~37.5 minutes training time
    Config.TOTAL_EPOCHS = 15
    Config.PHASE1_EPOCHS = 10  # Deep Supervision active
    # Phase 2 will run from epoch 10 to 14 (5 epochs)

    print(f"Starting Dynamic Deep Supervision Stratified Ensemble")
    print(f"Configuration: {Config.FOLDS} Folds, {Config.TOTAL_EPOCHS} Epochs/Fold")
    print(f"Device: {Config.DEVICE}")

    # Containers for Global OOF Analysis
    oof_preds_all = []
    oof_targets_all = []
    oof_ids_all = []
    oof_depths_all = []
    oof_coverages_all = []

    # 2. Cross-Validation Loop
    for fold in range(Config.FOLDS):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Data Loading
        train_loader, val_loader = get_dataloaders(fold=fold, load_cached_data=True)

        # Model Initialization
        model = SaltUNetPlusPlus(deep_supervision=True)
        model.to(Config.DEVICE)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.PHASE1_LR, weight_decay=Config.WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )

        # Engine
        engine = SaltEngine(model, optimizer, Config.DEVICE, scheduler)

        # Training Loop
        best_map = 0.0
        best_epoch = 0
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}_best.pth")

        for epoch in range(Config.TOTAL_EPOCHS):
            # Update LR for Phase 2 transition manually if needed,
            # though Scheduler handles plateau.
            # Strategy says explicit LR change for Phase 2:
            if epoch == Config.PHASE1_EPOCHS:
                print("Switching to Phase 2: Resetting LR for Fine-tuning")
                for param_group in optimizer.param_groups:
                    param_group["lr"] = Config.PHASE2_LR

            train_loss = engine.train_one_epoch(train_loader, epoch)
            val_map = engine.validate_one_epoch(val_loader)

            # Step scheduler
            if scheduler:
                scheduler.step(val_map)

            # Save Best
            if val_map > best_map:
                best_map = val_map
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
                print(f"New Best Score! Saved to {checkpoint_path}")

        print(f"Fold {fold} Finished. Best mAP: {best_map:.4f} at Epoch {best_epoch+1}")

        # Load Best Model for OOF Inference
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        model.eval()

        # Generate OOF Predictions
        # We need to manually iterate to get preds, targets, and metadata
        # Engine.predict gives preds, but we need targets too.
        print("Generating OOF predictions...")

        # Crop indices for validation (revert padding)
        pad_total = Config.IMG_HEIGHT - Config.ORIG_HEIGHT
        pad_top = pad_total // 2
        start_idx = pad_top
        end_idx = Config.IMG_HEIGHT - (pad_total - pad_top)

        with torch.no_grad():
            for images, masks, ids in val_loader:
                images = images.to(Config.DEVICE, dtype=torch.float32)

                # Forward
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Crop
                probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]

                # Handle masks (which might be padded in dataset, need to crop)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)  # (B, 1, H, W)
                masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]

                # Store
                preds_np = probs_cropped.cpu().numpy()[:, 0, :, :]
                targets_np = masks_cropped.cpu().numpy()[:, 0, :, :]

                for i in range(len(ids)):
                    oof_preds_all.append(preds_np[i])
                    oof_targets_all.append(targets_np[i])
                    oof_ids_all.append(ids[i])

                    # Retrieve metadata for failure analysis
                    # We can look this up from metadata dataframe later using ID
                    # but for now we will just store ID.

        # Cleanup
        del model, optimizer, scheduler, engine, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Global Threshold Optimization
    print("\n" + "=" * 40)
    print("Global Threshold Optimization")
    print("=" * 40)

    oof_preds_all = np.array(oof_preds_all)
    oof_targets_all = np.array(oof_targets_all)

    # Sweep thresholds for probability -> binary conversion
    # The metric calculates mAP over IoU thresholds 0.5:0.95
    # We want to find the best probability threshold to maximize this mAP.
    prob_thresholds = np.arange(0.3, 0.75, 0.05)
    best_global_map = 0.0
    best_prob_thresh = 0.5

    for t in prob_thresholds:
        # Binarize predictions
        preds_bin = (oof_preds_all > t).astype(np.uint8)

        # Calculate mAP
        # Note: calc_map expects probabilities usually, but if we pass binary
        # it treats them as hard predictions.
        # However, calc_map in utils thresholds at 0.5 internally.
        # So we should pass the probabilities but we want to simulate different thresholds.
        # The provided utils.calc_map hardcodes `predict > 0.5`.
        # To optimize threshold, we can shift the probabilities:
        # prob > thresh <=> (prob - thresh + 0.5) > 0.5

        shifted_preds = oof_preds_all - t + 0.5
        # Clip to 0-1 to be safe, though not strictly necessary for the >0.5 check
        shifted_preds = np.clip(shifted_preds, 0, 1)

        score = calc_map(shifted_preds, oof_targets_all)
        print(f"Threshold {t:.2f} -> mAP: {score:.5f}")

        if score > best_global_map:
            best_global_map = score
            best_prob_thresh = t

    print(f"\nFinal Validation Metric: {best_global_map:.10f}")
    print(f"Optimized Probability Threshold: {best_prob_thresh:.2f}")

    # 4. Failure Analysis
    print("\n" + "=" * 40)
    print("Failure Analysis")
    print("=" * 40)

    # Load metadata to link IDs to Depth and Coverage
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    full_meta = pd.concat([train_meta, val_meta], ignore_index=True)
    meta_map = full_meta.set_index("id")

    # Calculate per-image mAP at best threshold
    per_image_scores = []
    depths = []
    coverages = []

    # Shift preds for best threshold
    final_oof_preds = np.clip(oof_preds_all - best_prob_thresh + 0.5, 0, 1)

    for i in range(len(oof_ids_all)):
        pid = oof_ids_all[i]
        p = final_oof_preds[i : i + 1]  # Keep batch dim
        t = oof_targets_all[i : i + 1]

        score = calc_map(p, t)
        per_image_scores.append(score)

        # Get metadata
        if pid in meta_map.index:
            row = meta_map.loc[pid]
            depths.append(row["z"])
            coverages.append(row["coverage"])
        else:
            depths.append(0)
            coverages.append(0)

    # Calculate Correlations
    # Error = 1 - Score
    errors = 1.0 - np.array(per_image_scores)

    corr_depth, _ = pearsonr(errors, depths)
    corr_cov, _ = pearsonr(errors, coverages)

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 5. Submission
    if best_global_map > 0.827:
        print("\n" + "=" * 40)
        print("Generating Submission")
        print("=" * 40)

        test_loader = get_test_loader(load_cached_data=True)

        # Initialize ensemble accumulator
        # We don't know exact size yet, will init on first batch
        ensemble_preds = {}  # id -> accumulated_prob_map
        ensemble_counts = {}  # id -> count

        # Iterate over all 5 folds
        for fold in range(Config.FOLDS):
            print(f"Inference with Fold {fold} Model...")
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"fold_{fold}_best.pth"
            )

            model = SaltUNetPlusPlus(deep_supervision=False)  # DS off for inference
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=Config.DEVICE)
            )
            model.to(Config.DEVICE)

            engine = SaltEngine(model, None, Config.DEVICE)

            # Predict with TTA
            preds, ids = engine.predict(test_loader, tta=True)

            for pid, pmap in zip(ids, preds):
                if pid not in ensemble_preds:
                    ensemble_preds[pid] = np.zeros_like(pmap, dtype=np.float32)
                    ensemble_counts[pid] = 0
                ensemble_preds[pid] += pmap
                ensemble_counts[pid] += 1

            del model, engine
            torch.cuda.empty_cache()
            gc.collect()

        # Average and Save
        final_ids = []
        final_rles = []

        print("Encoding and Saving...")
        # Ensure order matches sample submission if possible, or just list all
        # The competition usually requires specific order or just ID matching.
        # We will process all keys.

        for pid in sorted(ensemble_preds.keys()):
            # Average
            avg_prob = ensemble_preds[pid] / ensemble_counts[pid]

            # Apply Optimized Threshold
            binary_mask = (avg_prob > best_prob_thresh).astype(np.uint8)

            # RLE Encode
            rle = rle_encode(binary_mask)

            final_ids.append(pid)
            final_rles.append(rle)

        submission_df = pd.DataFrame({"id": final_ids, "rle_mask": final_rles})
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {best_global_map:.4f} <= 0.827. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
