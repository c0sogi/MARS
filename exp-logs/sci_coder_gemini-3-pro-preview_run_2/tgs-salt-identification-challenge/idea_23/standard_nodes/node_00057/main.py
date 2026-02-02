import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Import library modules
from library.config import Config
from library.utils import set_seed, create_submission, calc_map_score, calculate_iou
from library.dataset import get_data_arrays, SaltDataset, get_transforms
from library.model import ResNet34WideLinkNet
from library.losses import CombinedLoss
from library.engine import train_one_epoch, train_student_epoch, validate, predict


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("\n=== Loading Data ===")
    # Load data arrays (cached if available)
    train_imgs, train_masks, train_depths, train_ids = get_data_arrays(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data=True
    )
    val_imgs, val_masks, val_depths, val_ids = get_data_arrays(
        Config.VAL_METADATA_PATH, "val", load_cached_data=True
    )
    test_imgs, _, test_depths, test_ids = get_data_arrays(
        Config.TEST_METADATA_PATH, "test", load_cached_data=True
    )

    # Calculate depth stats for standardization
    all_depths = np.concatenate([train_depths, val_depths])
    depth_mean = all_depths.mean()
    depth_std = all_depths.std()
    depth_stats = (depth_mean, depth_std)

    print(
        f"Train size: {len(train_imgs)}, Val size: {len(val_imgs)}, Test size: {len(test_imgs)}"
    )

    # ====================================================
    # Stage 1: Teacher Ensemble Training
    # ====================================================
    print("\n=== Stage 1: Teacher Ensemble Training ===")

    # Configuration for fast baseline
    n_folds = 5
    epochs_stage1 = 10  # Reduced for time constraint
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    teacher_models = []

    # We split the training data into folds
    for fold, (train_idx, _) in enumerate(kf.split(train_imgs)):
        print(f"\nTraining Teacher Fold {fold+1}/{n_folds}")

        # Subset data
        f_train_imgs = train_imgs[train_idx]
        f_train_masks = train_masks[train_idx]
        f_train_depths = train_depths[train_idx]
        f_train_ids = train_ids[train_idx]

        # Dataset & Loader
        # Note: depth_dropout_prob=0.5 for Teacher robustness
        train_ds = SaltDataset(
            f_train_imgs,
            f_train_masks,
            f_train_depths,
            f_train_ids,
            transforms=get_transforms("train"),
            depth_stats=depth_stats,
            depth_dropout_prob=0.5,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # Model, Optimizer, Loss
        model = ResNet34WideLinkNet().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        loss_fn = CombinedLoss()

        # Train Loop
        for epoch in range(epochs_stage1):
            loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
            # Simple logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs_stage1} - Loss: {loss:.4f}")

        # Save Teacher
        ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, f"teacher_fold_{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)
        teacher_models.append(model)

        # Clean up to save memory
        del train_ds, train_loader, optimizer
        torch.cuda.empty_cache()

    # ====================================================
    # Stage 2: Soft Pseudo-Label Generation
    # ====================================================
    print("\n=== Stage 2: Soft Pseudo-Label Generation ===")

    # Dataset for Test (No masks, depth dropout = 1.0 to force depth=0)
    # We use depth_dropout_prob=1.0 effectively setting all depths to mean (0)
    # This matches the teacher training condition where they saw 0 depth 50% of time.
    test_ds = SaltDataset(
        test_imgs,
        None,
        test_depths,
        test_ids,
        transforms=get_transforms("test"),
        depth_stats=depth_stats,
        depth_dropout_prob=1.0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Accumulate predictions
    avg_preds = np.zeros(
        (len(test_imgs), Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.float32
    )

    for i, model in enumerate(teacher_models):
        print(f"Predicting with Teacher {i+1}...")
        _, preds = predict(model, test_loader, device)

        # Crop back to 101x101 if predict returns 128x128 (predict usually handles logic,
        # but let's check shape. predict() in engine returns whatever model outputs.
        # The model outputs 128x128. We need to center crop to 101x101.
        # Actually, let's check engine.predict. It returns numpy array.
        # The model output is (B, 1, 128, 128).
        # We need to crop to 101x101 before averaging to save memory/compute.

        # Center crop logic
        if preds.shape[-1] == 128:
            start = (128 - 101) // 2
            end = start + 101
            preds = preds[:, start:end, start:end]

        avg_preds += preds

    avg_preds /= n_folds
    print("Pseudo-labels generated.")

    # Clean up teachers
    del teacher_models
    torch.cuda.empty_cache()

    # ====================================================
    # Stage 3: Student Training
    # ====================================================
    print("\n=== Stage 3: Student Training ===")

    # Labeled Loader (Full Train)
    labeled_ds = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transforms=get_transforms("train"),
        depth_stats=depth_stats,
        depth_dropout_prob=0.0,  # Student sees real depth for labeled data
    )
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Unlabeled Loader (Test with Soft Labels)
    # Student sees Unlabeled data with depth=0 (Test condition)
    unlabeled_ds = SaltDataset(
        test_imgs,
        avg_preds,
        test_depths,
        test_ids,
        transforms=get_transforms("student"),  # Strong Augmentation
        depth_stats=depth_stats,
        depth_dropout_prob=1.0,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Student Model
    student_model = ResNet34WideLinkNet().to(device)
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    loss_fn_labeled = CombinedLoss()
    loss_fn_unlabeled = torch.nn.BCEWithLogitsLoss()

    epochs_stage3 = 10

    for epoch in range(epochs_stage3):
        loss = train_student_epoch(
            student_model,
            labeled_loader,
            unlabeled_loader,
            optimizer,
            device,
            loss_fn_labeled,
            loss_fn_unlabeled,
        )
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs_stage3} - Loss: {loss:.4f}")

    # Save Student
    torch.save(
        student_model.state_dict(),
        os.path.join(Config.CHECKPOINTS_DIR, "student_final.pth"),
    )

    # ====================================================
    # Validation & Threshold Optimization
    # ====================================================
    print("\n=== Validation & Threshold Optimization ===")

    val_ds = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        transforms=get_transforms("valid"),
        depth_stats=depth_stats,
        depth_dropout_prob=0.0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Get raw predictions (probs)
    student_model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for imgs, masks, depths, _ in val_loader:
            imgs = imgs.to(device)
            depths = depths.to(device)

            logits = student_model(imgs, depths)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Crop to 101x101
            if probs.shape[-1] == 128:
                start = (128 - 101) // 2
                end = start + 101
                probs = probs[:, :, start:end, start:end]

            # Squeeze channel
            if probs.ndim == 4:
                probs = probs.squeeze(1)

            val_preds.append(probs)
            val_targets.append(masks.numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Threshold Search (maximizing mAP)
    best_thresh = 0.5
    best_score = 0.0

    # We search for the threshold that generates the best binary mask
    # The mAP metric itself sweeps 0.5-0.95, but our binary submission depends on one threshold.
    # Actually, the metric function takes binary inputs or logits.
    # If we pass probabilities, calc_map_score internally binarizes at 0.5.
    # To optimize the score, we want to find a threshold T such that (probs > T) yields best mAP.

    thresholds = np.arange(0.3, 0.75, 0.05)
    print("Searching for optimal threshold...")

    for t in thresholds:
        # Binarize predictions at t
        binary_preds = (val_preds > t).astype(np.uint8)
        # Calculate mAP (which sweeps IoU thresholds 0.5-0.95)
        score = calc_map_score(binary_preds, val_targets)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Final Validation Metric: {best_score}")

    # ====================================================
    # Failure Analysis
    # ====================================================
    print("\n=== Failure Analysis ===")
    ious = []
    for i in range(len(val_preds)):
        p = (val_preds[i] > best_thresh).astype(np.uint8)
        t = val_targets[i].astype(np.uint8)
        ious.append(calculate_iou(p, t))

    ious = np.array(ious)
    errors = 1.0 - ious

    # Features
    salt_coverage = np.array([m.sum() / m.size for m in val_targets])
    # Depth (need to align with val_loader order, but val_loader is sequential on val_ds)
    # val_ds uses val_depths directly
    depth_values = val_depths

    # Correlations
    corr_depth = np.corrcoef(errors, depth_values)[0, 1]
    corr_salt = np.corrcoef(errors, salt_coverage)[0, 1]

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_salt:.4f}")

    # ====================================================
    # Submission
    # ====================================================
    if best_score > 0.7985:
        print("\n=== Generating Submission ===")

        # Predict on Test
        # Use student model, depth=0 (handled by test_loader/ds config earlier)
        # Re-use test_loader from Stage 2 (it has depth_dropout_prob=1.0)

        test_ids_list, test_probs_raw = predict(student_model, test_loader, device)

        # Crop
        if test_probs_raw.shape[-1] == 128:
            start = (128 - 101) // 2
            end = start + 101
            test_probs_raw = test_probs_raw[:, start:end, start:end]

        # Binarize
        test_masks = (test_probs_raw > best_thresh).astype(np.uint8)

        # Create Submission
        create_submission(test_ids_list, test_masks, Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {best_score:.4f} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
