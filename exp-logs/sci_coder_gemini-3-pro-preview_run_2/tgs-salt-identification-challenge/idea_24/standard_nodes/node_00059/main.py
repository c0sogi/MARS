import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from itertools import cycle

# Import library modules
from library.config import Config
from library.utils import (
    load_data_with_cache,
    unpad_image,
    calc_map_score,
    create_submission,
    rle_encode,
)
from library.model import ResNet34WideLinkNet
from library.losses import MultiTaskLoss
from library.dataset import get_dataloaders, SaltDataset
from library.engine import set_seed, train_one_epoch, evaluate, predict

# -------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -------------------------------------------------------------------------
# To ensure execution within 2 hours, we reduce epochs.
# A100 is fast, but we want to be safe.
Config.STAGE1_EPOCHS = 12
Config.STAGE3_EPOCHS = 12
Config.BATCH_SIZE = 64  # Increase batch size for A100
Config.NUM_WORKERS = 4


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Create working directories
    Config.setup()

    # -------------------------------------------------------------------------
    # Stage 1: Train Teacher Model (Supervised)
    # -------------------------------------------------------------------------
    print("\n=== Stage 1: Training Teacher Model ===")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Initialize Model
    teacher_model = ResNet34WideLinkNet(pretrained=True).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    criterion = MultiTaskLoss(aux_weight=Config.AUX_DEPTH_LOSS_WEIGHT)

    # Training Loop
    best_val_loss = float("inf")
    best_teacher_path = os.path.join(Config.CHECKPOINT_DIR, "teacher_best.pth")

    for epoch in range(1, Config.STAGE1_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.STAGE1_EPOCHS}")
        train_loss, train_metrics = train_one_epoch(
            teacher_model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_map = evaluate(teacher_model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(teacher_model.state_dict(), best_teacher_path)
            print(
                f"  [Saved Best Teacher] Val Loss: {val_loss:.4f} | mAP: {val_map:.4f}"
            )

    # -------------------------------------------------------------------------
    # Stage 2: Generate Soft Pseudo-Labels
    # -------------------------------------------------------------------------
    print("\n=== Stage 2: Generating Soft Pseudo-Labels ===")

    # Load best teacher
    teacher_model.load_state_dict(torch.load(best_teacher_path, map_location=device))
    teacher_model.eval()

    # Predict on Test Set (using TTA from engine.predict)
    test_ids, test_probs = predict(teacher_model, test_loader, device)

    # test_probs is (N, 101, 101) unpadded.
    # We need to use these as targets for the Student.
    # The Student training pipeline expects padded images (128x128).
    # So we need to pad these masks back to 128x128 for the dataset.

    from library.utils import pad_image

    padded_soft_masks = []
    for i in range(len(test_probs)):
        # Pad back to 128x128
        p_mask = pad_image(test_probs[i], Config.IMG_H, Config.IMG_W)
        padded_soft_masks.append(p_mask)

    padded_soft_masks = np.array(padded_soft_masks, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Stage 3: Train Student Model (Semi-Supervised)
    # -------------------------------------------------------------------------
    print("\n=== Stage 3: Training Student Model ===")

    # Prepare Student Data
    # We need a custom dataset/loader that yields (image, soft_mask, depth)
    # Re-using SaltDataset logic but with soft masks

    # Load raw test images again to pass to dataset
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    # Ensure order matches test_ids from predict
    # predict returns ids in order of loader, which is order of df.

    # Load test images from cache
    test_images_raw = np.load(os.path.join(Config.CACHE_DIR, "test_images.npy"))
    test_depths_raw = test_df["z"].values.astype(np.float32)

    # Get depth stats from train for normalization
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    depth_mean = train_df["z"].mean()
    depth_std = train_df["z"].std() + 1e-6

    # Create Student Dataset (Pseudo)
    # Note: We apply training augmentations to the student's test data too!
    student_test_dataset = SaltDataset(
        ids=test_df["id"].values,
        images=test_images_raw,
        depths=test_depths_raw,
        masks=padded_soft_masks,  # Soft targets
        mode="pseudo",  # Treated like train/val but with soft masks
        transform=train_loader.dataset.transform,  # Use same augs as train
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    student_test_loader = DataLoader(
        student_test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize Student
    student_model = ResNet34WideLinkNet(pretrained=True).to(device)
    student_optimizer = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Losses
    criterion_labeled = MultiTaskLoss(aux_weight=Config.AUX_DEPTH_LOSS_WEIGHT)
    criterion_unlabeled = nn.BCEWithLogitsLoss()

    best_student_map = 0.0
    best_student_path = os.path.join(Config.CHECKPOINT_DIR, "student_best.pth")

    # Semi-Supervised Loop
    for epoch in range(1, Config.STAGE3_EPOCHS + 1):
        student_model.train()
        running_loss = 0.0
        steps = 0

        # Zip labeled and unlabeled data
        # Cycle the smaller dataset (usually test is smaller or similar)
        # Here train=2400, test=1000.
        loader_zip = zip(train_loader, cycle(student_test_loader))

        # Number of steps = length of larger loader (train)
        num_steps = len(train_loader)

        for i, (batch_labeled, batch_unlabeled) in enumerate(loader_zip):
            if i >= num_steps:
                break

            # --- Labeled Step ---
            imgs_l, masks_l, depths_l = batch_labeled
            imgs_l, masks_l, depths_l = (
                imgs_l.to(device),
                masks_l.to(device),
                depths_l.to(device),
            )

            # --- Unlabeled Step ---
            imgs_u, masks_u, _ = batch_unlabeled  # Ignore depth for unlabeled in loss
            imgs_u, masks_u = imgs_u.to(device), masks_u.to(device)

            student_optimizer.zero_grad()

            # Forward Labeled
            logits_l, pred_depths_l = student_model(imgs_l)
            loss_l, _ = criterion_labeled(logits_l, pred_depths_l, masks_l, depths_l)

            # Forward Unlabeled
            logits_u, _ = student_model(imgs_u)
            # masks_u are soft probabilities (0-1). BCEWithLogitsLoss accepts soft targets.
            loss_u = criterion_unlabeled(logits_u, masks_u)

            # Combined Loss (Equal weight)
            loss = loss_l + loss_u

            loss.backward()
            student_optimizer.step()

            running_loss += loss.item()
            steps += 1

        epoch_loss = running_loss / steps
        print(f"Epoch {epoch} Student Loss: {epoch_loss:.4f}")

        # Validation
        val_loss, val_map = evaluate(
            student_model, val_loader, criterion_labeled, device
        )

        if val_map > best_student_map:
            best_student_map = val_map
            torch.save(student_model.state_dict(), best_student_path)
            print(f"  [Saved Best Student] Val mAP: {val_map:.4f}")

    # -------------------------------------------------------------------------
    # Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Final Evaluation & Failure Analysis ===")

    # Load Best Student
    student_model.load_state_dict(torch.load(best_student_path, map_location=device))
    student_model.eval()

    # 1. Final Validation Metric
    # We need to compute it exactly as requested
    _, final_map = evaluate(student_model, val_loader, criterion_labeled, device)
    print(f"Final Validation Metric: {final_map}")

    # 2. Failure Analysis
    # Calculate per-image IoU and correlate with Depth
    print("Performing Failure Analysis...")

    val_ious = []
    val_depths_list = []

    with torch.no_grad():
        for batch in val_loader:
            images, masks, depths = batch
            images = images.to(device)

            # Predict
            logits, _ = student_model(images)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()
            depths_np = depths.numpy()

            # Unpad and calc IoU per image
            for i in range(len(images)):
                p = unpad_image(probs_np[i, 0], Config.ORIG_H, Config.ORIG_W)
                m = unpad_image(masks_np[i, 0], Config.ORIG_H, Config.ORIG_W)

                # Binarize at 0.5
                p_bin = (p > 0.5).astype(np.uint8)
                m_bin = (m > 0.5).astype(np.uint8)

                intersection = np.sum(p_bin & m_bin)
                union = np.sum(p_bin | m_bin)

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                val_ious.append(iou)

                # Denormalize depth for interpretation
                # depth_tensor was (z - mean) / std
                d_raw = (depths_np[i] * depth_std) + depth_mean
                val_depths_list.append(d_raw)

    val_ious = np.array(val_ious)
    val_depths_list = np.array(val_depths_list).flatten()

    # Calculate Error
    errors = 1.0 - val_ious

    # Correlation
    correlation = np.corrcoef(errors, val_depths_list)[0, 1]

    print(f"Failure Analysis Report:")
    print(f"  Mean IoU: {np.mean(val_ious):.4f}")
    print(f"  Mean Error: {np.mean(errors):.4f}")
    print(f"  Correlation (Error vs Depth): {correlation:.4f}")

    if abs(correlation) > 0.2:
        print(
            "  -> Significant correlation detected. Depth is a likely failure factor."
        )
    else:
        print("  -> Low correlation. Errors are likely depth-independent.")

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.7985

    if final_map > THRESHOLD_SCORE:
        print(
            f"\nValidation mAP ({final_map:.4f}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Generate prediction on Test Set
        # Use predict() which handles TTA and unpadding
        test_ids, test_probs = predict(student_model, test_loader, device)

        # Binarize
        binary_masks = (test_probs > 0.5).astype(np.uint8)

        # Save
        create_submission(test_ids, binary_masks, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation mAP ({final_map:.4f}) <= Threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
