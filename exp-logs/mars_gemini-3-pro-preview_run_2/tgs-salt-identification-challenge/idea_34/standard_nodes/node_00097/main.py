import sys
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import load_data, SaltDataset, get_transforms
from library.model import SpecialistTeacher, GeneralistStudent
from library.losses import LovaszHingeLoss, StudentMultiTaskLoss
from library.engine import (
    set_seed,
    fit_model,
    predict_marginalized,
    generate_submission,
    validate,
    do_kaggle_metric,
)
from library.utils import calculate_iou, optimize_threshold


def run_stage1_teacher(device, train_loader, val_loader):
    print("\n=== Stage 1: Training Specialist Teacher ===")

    model = SpecialistTeacher().to(device)

    # Loss: Lovasz + BCE
    bce_fn = nn.BCEWithLogitsLoss()
    lovasz_fn = LovaszHingeLoss()

    def criterion(logits, targets):
        return 0.5 * bce_fn(logits, targets) + 0.5 * lovasz_fn(logits, targets)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train
    epochs = 15

    best_metric = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=criterion,
        epochs=epochs,
        is_student=False,
        patience=5,
        save_path=os.path.join(Config.CHECKPOINT_DIR, "teacher_stage1.pth"),
    )

    print(f"Stage 1 Best Val mAP: {best_metric}")
    return model


def run_stage2_pseudo(device, teacher_model, test_loader):
    print("\n=== Stage 2: Generating Marginalized Pseudo-Labels ===")

    # Predict using marginalized inference
    # Returns dict {id: prob_map_numpy}
    pseudo_results = predict_marginalized(teacher_model, test_loader, device)

    return pseudo_results


def prepare_student_data(train_dataset, test_dataset, pseudo_results):
    print("Preparing Combined Dataset for Student...")

    # Extract Train Data
    train_imgs = train_dataset.images
    train_masks = train_dataset.masks
    train_depths = train_dataset.depths
    train_ids = train_dataset.ids

    # Extract Test Data & Pseudo Labels
    test_imgs = test_dataset.images
    test_depths = test_dataset.depths
    test_ids = test_dataset.ids

    # Align pseudo labels with test images
    test_masks_list = []
    valid_indices = []

    for i, img_id in enumerate(test_ids):
        if img_id in pseudo_results:
            # Pseudo label is (1, H, W) or (H, W) prob map
            mask = pseudo_results[img_id]
            if mask.ndim == 3:
                mask = mask.squeeze(0)  # Ensure (H, W) for Dataset compatibility

            test_masks_list.append(mask)
            valid_indices.append(i)

    test_masks = np.array(test_masks_list)

    # Ensure train masks are compatible (N, 128, 128)
    if train_masks.ndim == 4:
        train_masks = train_masks.squeeze(1)

    # Filter test images/depths
    test_imgs = test_imgs[valid_indices]
    test_depths = test_depths[valid_indices]
    test_ids = test_ids[valid_indices]

    # Concatenate
    combined_imgs = np.concatenate([train_imgs, test_imgs], axis=0)
    combined_masks = np.concatenate([train_masks, test_masks], axis=0)
    combined_depths = np.concatenate([train_depths, test_depths], axis=0)
    combined_ids = np.concatenate([train_ids, test_ids], axis=0)

    # Create Dataset
    student_dataset = SaltDataset(
        images=combined_imgs,
        masks=combined_masks,
        depths=combined_depths,
        ids=combined_ids,
        transform=get_transforms("train"),
        mode="train",
    )

    return student_dataset


def run_stage3_student(device, combined_dataset, val_loader):
    print("\n=== Stage 3: Training Generalist Student ===")

    train_loader = DataLoader(
        combined_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = GeneralistStudent().to(device)

    # Loss: MultiTask (BCE + MSE)
    criterion = StudentMultiTaskLoss(depth_weight=0.5)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    epochs = 15

    best_metric = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=criterion,
        epochs=epochs,
        is_student=True,
        patience=5,
        save_path=os.path.join(Config.CHECKPOINT_DIR, "student_stage3.pth"),
    )

    print(f"Stage 3 Best Val mAP: {best_metric}")
    return model


def failure_analysis(model, val_loader, device, is_student=True):
    print("\n=== Failure Analysis ===")
    model.eval()

    ious = []
    depths_list = []
    salt_coverages = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks_gpu = masks.to(device, dtype=torch.float32)
            depths_gpu = depths.to(device, dtype=torch.float32)

            if is_student:
                logits, _ = model(images)
            else:
                logits = model(images, depths_gpu)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Calculate IoU per image
            for i in range(len(images)):
                p = preds[i].cpu().numpy().flatten()
                t = masks_gpu[i].cpu().numpy().flatten()

                intersection = np.logical_and(p, t).sum()
                union = np.logical_or(p, t).sum()
                iou = intersection / union if union > 0 else 1.0

                ious.append(iou)
                depths_list.append(depths_gpu[i].item())
                salt_coverages.append(t.mean())

    ious = np.array(ious)
    depths_list = np.array(depths_list)
    salt_coverages = np.array(salt_coverages)

    # Correlations (Error = 1 - IoU)
    error = 1.0 - ious

    corr_depth = np.corrcoef(error, depths_list)[0, 1]
    if np.std(salt_coverages) > 0:
        corr_salt = np.corrcoef(error, salt_coverages)[0, 1]
    else:
        corr_salt = 0.0

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_salt:.4f}")

    return ious.mean()


def main():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = load_data("train", load_cached_data=True)
    val_dataset = load_data("val", load_cached_data=True)
    test_dataset = load_data("test", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Stage 1: Teacher
    teacher_model = run_stage1_teacher(device, train_loader, val_loader)

    # 3. Stage 2: Pseudo-Labeling
    pseudo_results = run_stage2_pseudo(device, teacher_model, test_loader)

    # 4. Prepare Student Data
    combined_dataset = prepare_student_data(train_dataset, test_dataset, pseudo_results)

    # 5. Stage 3: Student
    student_model = run_stage3_student(device, combined_dataset, val_loader)

    # 6. Final Validation & Threshold Optimization
    print("\n=== Final Validation ===")

    student_model.eval()
    val_preds = []
    val_truths = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(device, dtype=torch.float32)

            # Student inference (no depth input)
            logits, _ = student_model(images)
            probs = torch.sigmoid(logits)

            # TTA
            if Config.TTA_ENABLED:
                images_flip = torch.flip(images, [3])
                logits_flip, _ = student_model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, [3])
                probs = (probs + probs_flip) / 2.0

            val_preds.append(probs.cpu().numpy())
            val_truths.append(masks.numpy())

    val_preds = np.concatenate(val_preds, axis=0).squeeze()
    val_truths = np.concatenate(val_truths, axis=0).squeeze()

    # Optimize Threshold
    best_threshold = optimize_threshold(val_preds, val_truths)
    print(f"Optimal Threshold: {best_threshold:.4f}")

    # Calculate Final Metric
    final_metric = do_kaggle_metric(val_preds, val_truths, threshold=best_threshold)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    failure_analysis(student_model, val_loader, device, is_student=True)

    # 8. Submission
    if final_metric > 0.7985:
        print("\nGenerating Submission...")
        generate_submission(
            student_model, test_loader, device, threshold=best_threshold
        )
    else:
        print(f"\nMetric {final_metric} <= 0.7985. Skipping submission.")


if __name__ == "__main__":
    main()
