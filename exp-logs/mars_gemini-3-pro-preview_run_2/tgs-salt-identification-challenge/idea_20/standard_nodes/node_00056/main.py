import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2

# Import from provided library files
from library.utils import set_seed, rle_encode, calc_map_score
from library.model import ResNet34WideLinkNet
from library.dataset import (
    get_loaders,
    preprocess_and_cache,
    SaltDataset,
    get_transforms,
)
from library.engine import train_one_epoch, evaluate, predict_proba
from library.losses import CombinedLoss

# --- Configuration ---
SEED = 42
BATCH_SIZE = 32
LR = 1e-3
EPOCHS_STAGE_1 = 20
EPOCHS_STAGE_2 = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBMISSION_THRESHOLD = 0.7985
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # 2. Data Loading (Stage 1)
    print("\n--- Loading Data for Stage 1 ---")
    # We use the standard loaders for the teacher training
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # 3. Stage 1: Teacher Training
    print("\n--- Stage 1: Teacher Training (Supervised with Depth Masking) ---")
    teacher_model = ResNet34WideLinkNet().to(DEVICE)
    optimizer = optim.AdamW(teacher_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE_1)

    best_teacher_map = 0.0
    best_teacher_path = os.path.join(WORKING_DIR, "teacher_best.pth")

    for epoch in range(1, EPOCHS_STAGE_1 + 1):
        loss = train_one_epoch(teacher_model, train_loader, optimizer, DEVICE, epoch)
        val_loss, val_map = evaluate(teacher_model, val_loader, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch}/{EPOCHS_STAGE_1} - Loss: {loss:.4f} - Val Map: {val_map:.4f}"
        )

        if val_map > best_teacher_map:
            best_teacher_map = val_map
            torch.save(teacher_model.state_dict(), best_teacher_path)

    print(f"Stage 1 Complete. Best Teacher mAP: {best_teacher_map:.4f}")

    # 4. Pseudo-Label Generation
    print("\n--- Generating Pseudo-Labels for Test Set ---")
    # Load best teacher
    teacher_model.load_state_dict(torch.load(best_teacher_path))

    # Generate soft predictions (predict_proba forces depth=0)
    # We need a loader for the test set that returns images and ids
    # The get_loaders function already returned test_loader
    pseudo_labels_dict = predict_proba(teacher_model, test_loader, DEVICE)
    print(f"Generated pseudo-labels for {len(pseudo_labels_dict)} test images.")

    # 5. Stage 2: Student Training (Combined Data)
    print("\n--- Stage 2: Student Training (Soft-Self-Training) ---")

    # Construct Combined Dataset Manually
    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Get Cached Arrays
    t_imgs, t_masks, t_depths, t_ids = preprocess_and_cache(
        train_df, "train", load_cached_data=True
    )
    v_imgs, v_masks, v_depths, v_ids = preprocess_and_cache(
        pd.read_csv("./metadata/val.csv"), "val", load_cached_data=True
    )
    test_imgs, _, test_depths, test_ids = preprocess_and_cache(
        test_df, "test", load_cached_data=True
    )

    # Concatenate Train and Test for the Student
    # Create dummy masks for test data (will be overridden by pseudo_labels logic in SaltDataset)
    test_masks_dummy = np.zeros_like(test_imgs)

    combined_imgs = np.concatenate([t_imgs, test_imgs])
    combined_masks = np.concatenate([t_masks, test_masks_dummy])
    combined_depths = np.concatenate([t_depths, test_depths])
    combined_ids = np.concatenate([t_ids, test_ids])

    # Calculate depth stats from original train set for normalization
    depth_mean = t_depths.mean()
    depth_std = t_depths.std()

    # Create Combined Dataset
    combined_ds = SaltDataset(
        combined_imgs,
        combined_masks,
        combined_depths,
        combined_ids,
        transform=get_transforms("train"),
        depth_stats=(depth_mean, depth_std),
        pseudo_labels=pseudo_labels_dict,
    )

    combined_loader = DataLoader(
        combined_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize Student
    student_model = ResNet34WideLinkNet().to(DEVICE)
    optimizer = optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE_2)

    best_student_map = 0.0
    best_student_path = os.path.join(WORKING_DIR, "student_best.pth")

    for epoch in range(1, EPOCHS_STAGE_2 + 1):
        # train_one_epoch handles the logic: if ID in pseudo_labels -> Soft Loss & Depth=0
        loss = train_one_epoch(student_model, combined_loader, optimizer, DEVICE, epoch)
        # Validate on original validation set
        val_loss, val_map = evaluate(student_model, val_loader, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch}/{EPOCHS_STAGE_2} - Loss: {loss:.4f} - Val Map: {val_map:.4f}"
        )

        if val_map > best_student_map:
            best_student_map = val_map
            torch.save(student_model.state_dict(), best_student_path)

    print(f"Stage 2 Complete. Best Student mAP: {best_student_map:.4f}")

    # 6. Threshold Optimization & Final Validation
    print("\n--- Threshold Optimization ---")
    student_model.load_state_dict(torch.load(best_student_path))
    student_model.eval()

    # Get raw probabilities for validation set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            # Use standard inference (depth included) for validation
            logits = student_model(images, depths)
            probs = torch.sigmoid(logits).cpu().numpy()

            val_probs.append(probs)
            val_targets.append(masks.numpy())

    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)

    # Search thresholds
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_thresh = 0.5
    best_score = 0.0

    for t in thresholds:
        # Binarize
        preds_bin = (val_probs > t).astype(np.uint8)
        # Calculate metric
        # calc_map_score expects (B, H, W)
        if preds_bin.ndim == 4:
            preds_bin = preds_bin[:, 0, :, :]
        if val_targets.ndim == 4:
            val_targets = val_targets[:, 0, :, :]

        score = calc_map_score(preds_bin, val_targets)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Final Validation Metric: {best_score:.10f}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-image mAP
    per_image_errors = []
    meta_depths = []
    meta_coverage = []

    # We need to match predictions to metadata.
    # val_loader order is fixed (shuffle=False).
    # val_probs and val_targets are aligned.

    # Load val metadata to get coverage info
    val_meta = pd.read_csv("./metadata/val.csv")

    for i in range(len(val_probs)):
        p = (val_probs[i] > best_thresh).astype(np.uint8)
        t = val_targets[i].astype(np.uint8)

        # calc_map_score handles batch, let's do single image logic manually or wrap
        # Reuse calc_map_score but for single item
        score = calc_map_score(p[np.newaxis, ...], t[np.newaxis, ...])
        error = 1.0 - score

        per_image_errors.append(error)

        # Get metadata
        row = val_meta.iloc[i]
        meta_depths.append(row["z"])
        meta_coverage.append(row["salt_coverage"])

    # Correlation
    err_arr = np.array(per_image_errors)
    depth_arr = np.array(meta_depths)
    cov_arr = np.array(meta_coverage)

    corr_depth = np.corrcoef(err_arr, depth_arr)[0, 1]
    corr_cov = np.corrcoef(err_arr, cov_arr)[0, 1]

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 8. Submission
    if best_score > SUBMISSION_THRESHOLD:
        print("\n--- Generating Submission ---")
        # Predict on Test Set (Force Depth=0)
        # We use predict_proba which already does TTA and Depth=0
        test_preds_dict = predict_proba(student_model, test_loader, DEVICE)

        submission_rows = []

        # Iterate through test_df to ensure order
        test_df = pd.read_csv("./metadata/test.csv")

        for idx, row in test_df.iterrows():
            img_id = row["id"]
            if img_id in test_preds_dict:
                prob_map = test_preds_dict[img_id]  # (128, 128)

                # Binarize
                mask = (prob_map > best_thresh).astype(np.uint8)

                # Crop back to 101x101
                # Padding was symmetric reflection.
                # 128 - 101 = 27.
                # pad_top = 13, pad_bottom = 14 (13+14=27)
                # pad_left = 13, pad_right = 14

                h, w = 101, 101
                target_size = 128
                pad_h = target_size - h
                pad_w = target_size - w
                pad_top = pad_h // 2
                pad_left = pad_w // 2

                mask_cropped = mask[pad_top : pad_top + h, pad_left : pad_left + w]

                # Encode
                rle = rle_encode(mask_cropped)
                submission_rows.append([img_id, rle])
            else:
                # Should not happen
                submission_rows.append([img_id, ""])

        sub_df = pd.DataFrame(submission_rows, columns=["id", "rle_mask"])
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Validation score {best_score:.4f} did not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
