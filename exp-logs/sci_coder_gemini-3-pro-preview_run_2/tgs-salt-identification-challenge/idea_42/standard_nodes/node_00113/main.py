import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
from library.config import Config
from library.utils import set_seed, get_iou_vector, do_kaggle_metric
from library.dataset import load_dataset_arrays, SaltDataset, get_transforms
from library.models import build_model
from library.engine import (
    train_one_epoch,
    evaluate,
    generate_pseudo_labels,
    predict_and_submit,
)


def main():
    # -------------------------------------------------------------------------
    # 0. Setup & Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    # Override Config for strict time limit execution (1 minute constraint)
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE3 = 1
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for speed
    Config.BATCH_SIZE = 16

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Starting FP32-Stabilized Marginalized-Distillation Pipeline...")
    print(f"Device: {device}")
    print(
        f"Debug Mode: {Config.DEBUG_SAMPLE_SIZE} samples, {Config.EPOCHS_STAGE1} epochs"
    )

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading Data...")

    # Load Train Data
    train_imgs, train_masks, train_depths, train_ids = load_dataset_arrays(
        Config.TRAIN_METADATA_PATH,
        cache_prefix="train",
        load_cached_data=True,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Load Val Data
    val_imgs, val_masks, val_depths, val_ids = load_dataset_arrays(
        Config.VAL_METADATA_PATH,
        cache_prefix="val",
        load_cached_data=True,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Load Test Data (for Pseudo-Labeling)
    test_imgs, _, test_depths, test_ids = load_dataset_arrays(
        Config.TEST_METADATA_PATH,
        cache_prefix="test",
        load_cached_data=True,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )
    # Create dummy masks for test set (needed for Dataset init, but unused)
    test_masks = np.zeros_like(test_imgs)

    # Create DataLoaders
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transforms=get_transforms("train"),
        mode="train",
    )
    val_dataset = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        transforms=get_transforms("val"),
        mode="val",
    )
    test_dataset = SaltDataset(
        test_imgs,
        test_masks,
        test_depths,
        test_ids,
        transforms=get_transforms("test"),
        mode="test",
    )

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

    # -------------------------------------------------------------------------
    # 2. Stage 1: Train Specialist Teacher
    # -------------------------------------------------------------------------
    print("\n[2/5] Stage 1: Training Specialist Teacher (Depth-Injected)...")

    teacher_model = build_model("teacher").to(device)
    optimizer_t = AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_t = CosineAnnealingLR(optimizer_t, T_max=Config.EPOCHS_STAGE1)

    for epoch in range(1, Config.EPOCHS_STAGE1 + 1):
        loss = train_one_epoch(
            teacher_model, train_loader, optimizer_t, device, epoch, mode="teacher"
        )
        scheduler_t.step()
        print(f"  Epoch {epoch}/{Config.EPOCHS_STAGE1} | Loss: {loss:.4f}")

    # Evaluate Teacher
    t_score, t_loss, _, _ = evaluate(teacher_model, val_loader, device, mode="teacher")
    print(f"  Teacher Validation mAP: {t_score:.4f}")

    # -------------------------------------------------------------------------
    # 3. Stage 2: Marginalized Pseudo-Labeling
    # -------------------------------------------------------------------------
    print("\n[3/5] Stage 2: Generating Marginalized Pseudo-Labels...")

    # We use the single trained teacher for this baseline (in full solution, use ensemble)
    pseudo_labels = generate_pseudo_labels(
        [teacher_model], test_loader, device, load_cached_data=False
    )

    # Create Pseudo-Labeled Dataset
    pseudo_dataset = SaltDataset(
        test_imgs,
        test_masks,
        test_depths,
        test_ids,
        transforms=get_transforms("train"),
        mode="pseudo",
        soft_labels=pseudo_labels,
    )

    # Combine Labeled + Pseudo-Labeled Data
    combined_dataset = ConcatDataset([train_dataset, pseudo_dataset])
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 4. Stage 3: Train Generalist Student
    # -------------------------------------------------------------------------
    print("\n[4/5] Stage 3: Training Generalist Student (Multi-Task)...")

    student_model = build_model("student").to(device)
    optimizer_s = AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_s = CosineAnnealingLR(optimizer_s, T_max=Config.EPOCHS_STAGE3)

    for epoch in range(1, Config.EPOCHS_STAGE3 + 1):
        loss = train_one_epoch(
            student_model, combined_loader, optimizer_s, device, epoch, mode="student"
        )
        scheduler_s.step()
        print(f"  Epoch {epoch}/{Config.EPOCHS_STAGE3} | Loss: {loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[5/5] Final Evaluation & Failure Analysis...")

    # Evaluate Student
    val_score, val_loss, val_preds, val_targets = evaluate(
        student_model, val_loader, device, mode="student"
    )

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {val_score}")

    # Failure Analysis
    # Calculate IoU per image
    val_preds_bin = (val_preds > 0.5).astype(np.uint8)
    ious = get_iou_vector(val_preds_bin, val_targets)
    errors = 1.0 - ious

    # Calculate Salt Coverage per image
    # val_targets is (N, H, W)
    pixel_counts = val_targets.reshape(val_targets.shape[0], -1).sum(axis=1)
    total_pixels = val_targets.shape[1] * val_targets.shape[2]
    coverages = pixel_counts / total_pixels

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {"error": errors, "depth": val_depths, "coverage": coverages}
    )

    corr_depth = df_analysis["error"].corr(df_analysis["depth"])
    corr_cov = df_analysis["error"].corr(df_analysis["coverage"])

    print("\nFailure Analysis:")
    print(f"  Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"  Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    SUBMISSION_THRESHOLD = 0.7985

    if val_score > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation score ({val_score:.4f}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # If we are in debug mode, we might not have the full test set loaded.
        # Ideally, we reload the full test set here, but for the fast baseline, we use what we have.
        # If Config.DEBUG_SAMPLE_SIZE is set, this submission will be partial,
        # but the logic remains valid for the full run.

        predict_and_submit(student_model, test_loader, val_loader, device)
    else:
        print(
            f"\nValidation score ({val_score:.4f}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
