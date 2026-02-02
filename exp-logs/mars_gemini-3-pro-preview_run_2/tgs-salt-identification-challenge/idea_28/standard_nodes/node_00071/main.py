import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import (
    prepare_train_data,
    prepare_test_data,
    get_dataloaders,
    get_test_dataloader,
    SaltDataset,
)
from library.model import SaltNet
from library.losses import MultiTaskLoss
from library.engine import (
    set_seed,
    train_one_epoch,
    evaluate,
    predict_tta,
    generate_submission,
)
from library.utils import optimize_threshold, calc_iou, get_score


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Set fast baseline parameters
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32

    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting Run. Device: {device}, Epochs per stage: {Config.EPOCHS}")

    # Ensure data is prepared and cached
    # This generates 'train_data_processed.parquet' with fold info
    train_meta_df = prepare_train_data(load_cached_data=True)
    test_meta_df = prepare_test_data(load_cached_data=True)

    # Create directories for artifacts
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Stage 1: Train 5-Fold Ensemble
    # -------------------------------------------------------------------------
    print("\n=== Stage 1: Training 5-Fold Ensemble ===")

    fold_models = []
    fold_metrics = []

    for fold in range(Config.NUM_FOLDS):
        print(f"\nTraining Fold {fold}/{Config.NUM_FOLDS - 1}")

        # Get dataloaders for this fold
        train_loader, val_loader = get_dataloaders(fold=fold, load_cached_data=True)

        # Initialize Model, Loss, Optimizer
        model = SaltNet().to(device)
        criterion = MultiTaskLoss(depth_weight=Config.DEPTH_LOSS_WEIGHT)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

        best_map = 0.0
        best_model_path = os.path.join(model_dir, f"model_fold_{fold}.pth")

        # Training Loop
        for epoch in range(1, Config.EPOCHS + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            # Save best model
            if val_metrics["map"] > best_map:
                best_map = val_metrics["map"]
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} Best mAP: {best_map:.4f}")
        fold_metrics.append(best_map)
        fold_models.append(best_model_path)

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. Stage 2: Quality Gating & Pseudo-Label Generation
    # -------------------------------------------------------------------------
    print("\n=== Stage 2: Quality Gating & Pseudo-Labeling ===")

    # Filter models based on validation performance
    valid_model_paths = []
    for fold, score in enumerate(fold_metrics):
        if score >= Config.MAP_THRESHOLD:
            valid_model_paths.append(fold_models[fold])
            print(f"Fold {fold} accepted (mAP {score:.4f} >= {Config.MAP_THRESHOLD})")
        else:
            print(f"Fold {fold} discarded (mAP {score:.4f} < {Config.MAP_THRESHOLD})")

    # Fallback if all models fail threshold
    if not valid_model_paths:
        print(
            "Warning: No models passed the threshold. Using the best performing model."
        )
        best_idx = np.argmax(fold_metrics)
        valid_model_paths.append(fold_models[best_idx])

    # Generate Pseudo-Labels on Test Set
    print("Generating pseudo-labels on Test set...")
    test_loader = get_test_dataloader(load_cached_data=True)

    avg_preds = None
    test_ids = []

    # Ensemble prediction
    for model_path in valid_model_paths:
        model = SaltNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Predict with TTA
        preds, ids = predict_tta(model, test_loader, device)

        if avg_preds is None:
            avg_preds = preds
            test_ids = ids
        else:
            avg_preds += preds

        del model
        torch.cuda.empty_cache()

    # Average probabilities
    avg_preds /= len(valid_model_paths)

    # Create dictionary mapping ID to Soft Mask (H, W)
    # avg_preds is (N, 1, H, W), we need (H, W)
    pseudo_labels = {}
    avg_preds_sq = avg_preds[:, 0, :, :]

    for idx, img_id in enumerate(test_ids):
        pseudo_labels[img_id] = avg_preds_sq[idx]

    print(f"Generated {len(pseudo_labels)} pseudo-labels.")

    # -------------------------------------------------------------------------
    # 4. Stage 3: Student Training
    # -------------------------------------------------------------------------
    print("\n=== Stage 3: Student Training ===")

    # Identify Hold-Out Validation Set
    val_csv_df = pd.read_csv(Config.VAL_CSV)
    val_ids = set(val_csv_df["id"].values)

    # Split processed train data into Student Train and Student Val
    # Student Train = All Labeled Data - HoldOut Val
    student_train_df = train_meta_df[~train_meta_df["id"].isin(val_ids)].reset_index(
        drop=True
    )
    student_val_df = train_meta_df[train_meta_df["id"].isin(val_ids)].reset_index(
        drop=True
    )

    print(
        f"Student Training Data: {len(student_train_df)} labeled + {len(test_meta_df)} pseudo"
    )
    print(f"Student Validation Data: {len(student_val_df)} (Hold-out)")

    # Construct Datasets
    ds_labeled = SaltDataset(student_train_df, mode="train")
    ds_pseudo = SaltDataset(
        test_meta_df, mode="pseudo_train", pseudo_labels=pseudo_labels
    )
    ds_train_combined = ConcatDataset([ds_labeled, ds_pseudo])

    ds_val = SaltDataset(student_val_df, mode="val")

    # Construct Loaders
    student_train_loader = DataLoader(
        ds_train_combined,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    student_val_loader = DataLoader(
        ds_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Student Model
    student_model = SaltNet().to(device)
    criterion = MultiTaskLoss(depth_weight=Config.DEPTH_LOSS_WEIGHT)
    optimizer = optim.AdamW(student_model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    best_student_map = 0.0
    best_student_path = os.path.join(model_dir, "student_best.pth")

    # Student Training Loop
    for epoch in range(1, Config.EPOCHS + 1):
        train_metrics = train_one_epoch(
            student_model, student_train_loader, criterion, optimizer, device, epoch
        )
        val_metrics = evaluate(student_model, student_val_loader, criterion, device)
        scheduler.step()

        if val_metrics["map"] > best_student_map:
            best_student_map = val_metrics["map"]
            torch.save(student_model.state_dict(), best_student_path)

    print(f"Student Best mAP: {best_student_map:.4f}")

    # Load best student model for final evaluation
    student_model.load_state_dict(torch.load(best_student_path, map_location=device))
    student_model.eval()

    # -------------------------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Final Validation & Failure Analysis ===")

    all_preds = []
    all_targets = []
    all_ids = []

    # Crop indices for metric calculation (128 -> 101)
    start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    end = start + Config.ORIG_SIZE

    # Inference on Hold-Out Set with TTA
    with torch.no_grad():
        for batch in student_val_loader:
            if len(batch) == 4:
                images, masks, depths, ids = batch
            else:
                continue

            images = images.to(device)
            masks = masks.to(device)

            # Original
            logits, _ = student_model(images)
            probs = torch.sigmoid(logits)

            # Flip TTA
            images_flip = torch.flip(images, dims=[3])
            logits_flip, _ = student_model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average
            avg_probs = (probs + probs_flip) / 2.0

            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Squeeze channel dimension
    if all_preds.ndim == 4:
        all_preds = all_preds[:, 0, :, :]
    if all_targets.ndim == 4:
        all_targets = all_targets[:, 0, :, :]

    # Crop to original size
    preds_crop = all_preds[:, start:end, start:end]
    targets_crop = all_targets[:, start:end, start:end]

    # Optimize Threshold
    best_th, best_score = optimize_threshold(preds_crop, targets_crop)
    print(f"Optimal Threshold: {best_th}, Score: {best_score:.5f}")

    # Print Final Metric
    final_metric = best_score
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    # Calculate error per image (1 - IoU)
    errors = []
    for i in range(len(preds_crop)):
        p = (preds_crop[i] > best_th).astype(np.uint8)
        t = targets_crop[i].astype(np.uint8)
        iou = calc_iou(p, t)
        errors.append(1.0 - iou)

    # Map errors back to metadata
    id_to_error = dict(zip(all_ids, errors))
    analysis_df = student_val_df.copy()
    analysis_df["error"] = analysis_df["id"].map(id_to_error)
    analysis_df = analysis_df.dropna(subset=["error"])

    # Correlations
    corr_depth = analysis_df["z"].corr(analysis_df["error"])
    corr_salt = analysis_df["salt_coverage"].corr(analysis_df["error"])

    print("\nFailure Analysis:")
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_salt:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.7985:
        print("\nMetric condition met. Generating submission...")
        generate_submission(
            student_model,
            test_loader,
            device,
            output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
            threshold=best_th,
        )
    else:
        print(f"\nMetric {final_metric:.4f} <= 0.7985. Skipping submission.")


if __name__ == "__main__":
    main()
