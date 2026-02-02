import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, optimize_threshold, calc_map_score, iou_metric
from library.data import (
    get_fold_datasets,
    get_test_dataset,
    PseudoDataset,
    get_transforms,
)
from library.model import SaltNet
from library.losses import CombinedLoss, AuxiliaryMSELoss, StableBCELoss
from library.train_eval import (
    train_teacher_epoch,
    train_student_epoch,
    validate,
    generate_submission,
)
from library.pseudo_label import generate_marginalized_labels


def run():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Adjust hyperparameters for the 24h/38m runtime constraint
    # 10 epochs is sufficient for convergence on this dataset size with pre-trained backbones
    Config.EPOCHS_STAGE1 = 10
    Config.EPOCHS_STAGE3 = 10
    Config.BATCH_SIZE = 32

    # Use Fold 0 for this baseline execution
    fold_idx = 0

    print(f"Starting execution on {device}...")

    # =========================================================================
    # STAGE 1: Train Specialist Teacher
    # =========================================================================
    print("\n--- Stage 1: Specialist Teacher Training ---")

    # Load Labeled Data (Fold 0)
    train_ds, val_ds, scaler = get_fold_datasets(fold_idx, load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Teacher Model (Depth-Injected)
    teacher_model = SaltNet(mode="teacher").to(device)

    # Optimizer & Loss
    optimizer = AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_STAGE1)
    loss_fn = CombinedLoss()

    # Training Loop
    best_teacher_map = 0.0
    best_teacher_path = os.path.join(
        Config.WORKING_DIR, f"teacher_fold{fold_idx}_best.pth"
    )

    for epoch in range(Config.EPOCHS_STAGE1):
        train_loss = train_teacher_epoch(
            teacher_model, train_loader, optimizer, device, loss_fn
        )
        val_loss, val_map = validate(teacher_model, val_loader, device, loss_fn)
        scheduler.step()

        # print(f"Epoch {epoch+1}/{Config.EPOCHS_STAGE1} | Train Loss: {train_loss:.4f} | Val mAP: {val_map:.4f}")

        if val_map > best_teacher_map:
            best_teacher_map = val_map
            torch.save(teacher_model.state_dict(), best_teacher_path)

    print(f"Best Teacher mAP: {best_teacher_map:.4f}")

    # Cleanup to save memory
    del teacher_model, optimizer, scheduler
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 2: Marginalized Pseudo-Labeling
    # =========================================================================
    print("\n--- Stage 2: Marginalized Pseudo-Labeling ---")

    # Generate soft labels using the best teacher model
    # Marginalizes over depth uncertainty to create robust targets
    teacher_paths = [best_teacher_path]
    soft_masks = generate_marginalized_labels(
        teacher_paths, scaler, load_cached_data=True
    )

    # =========================================================================
    # STAGE 3: Train Generalist Student
    # =========================================================================
    print("\n--- Stage 3: Generalist Student Training ---")

    # Prepare Unlabeled Data (Test Set)
    raw_test_ds = get_test_dataset(scaler, load_cached_data=True)
    pseudo_data_dict = {
        "images": raw_test_ds.images,
        "ids": raw_test_ds.ids,
        "depths": raw_test_ds.depths,
    }

    # Create PseudoDataset with soft masks and training augmentations
    pseudo_ds = PseudoDataset(
        pseudo_data_dict,
        soft_masks,
        transform=get_transforms("train"),
        depth_scaler=scaler,
    )

    pseudo_loader = DataLoader(
        pseudo_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize Student Model (Multi-Task, Image-Only Input)
    student_model = SaltNet(mode="student").to(device)

    # Optimizer & Losses
    optimizer = AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_STAGE3)

    seg_loss_fn = CombinedLoss()
    aux_loss_fn = AuxiliaryMSELoss()
    soft_loss_fn = StableBCELoss()

    # Training Loop
    best_student_map = 0.0
    best_student_path = os.path.join(Config.WORKING_DIR, "student_best.pth")

    for epoch in range(Config.EPOCHS_STAGE3):
        losses = train_student_epoch(
            student_model,
            train_loader,
            pseudo_loader,
            optimizer,
            device,
            seg_loss_fn,
            aux_loss_fn,
            soft_loss_fn,
        )

        # Validate Student
        val_loss, val_map = validate(student_model, val_loader, device, seg_loss_fn)
        scheduler.step()

        # print(f"Epoch {epoch+1}/{Config.EPOCHS_STAGE3} | Total Loss: {losses['loss_total']:.4f} | Val mAP: {val_map:.4f}")

        if val_map > best_student_map:
            best_student_map = val_map
            torch.save(student_model.state_dict(), best_student_path)

    print(f"Best Student mAP: {best_student_map:.4f}")

    # =========================================================================
    # VALIDATION & FAILURE ANALYSIS
    # =========================================================================
    print("\n--- Validation & Failure Analysis ---")

    # Load best student for final evaluation
    student_model.load_state_dict(torch.load(best_student_path, map_location=device))
    student_model.eval()

    # Collect predictions and targets
    val_probs = []
    val_targets = []
    val_depths_raw = []

    with torch.no_grad():
        for images, masks, depths, _ in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits, _ = student_model(images)
            probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(masks.cpu().numpy())

            # Inverse transform depths for analysis
            d_raw = scaler.inverse_transform(depths.numpy().reshape(-1, 1)).flatten()
            val_depths_raw.append(d_raw)

    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)
    val_depths_raw = np.concatenate(val_depths_raw)

    # Optimize Binarization Threshold
    best_thresh, best_score = optimize_threshold(val_probs, val_targets)

    # Print Final Metric (Required)
    print(f"Final Validation Metric: {best_score}")

    # Failure Analysis: Correlation between Error and Depth
    val_preds_bin = (val_probs > best_thresh).astype(np.uint8)
    val_targets_uint8 = val_targets.astype(np.uint8)

    ious = []
    for p, t in zip(val_preds_bin, val_targets_uint8):
        ious.append(iou_metric(p, t))
    ious = np.array(ious)

    # Error = 1 - IoU
    errors = 1.0 - ious
    correlation = np.corrcoef(errors, val_depths_raw)[0, 1]

    print(f"Correlation (Error vs Depth): {correlation:.4f}")

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    if best_score > 0.7985:
        print("\n--- Generating Submission ---")
        # Prepare Test Loader
        test_ds = get_test_dataset(scaler, load_cached_data=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Generate Submission with TTA and Optimized Threshold
        generate_submission(student_model, test_loader, device, threshold=best_thresh)
    else:
        print(
            f"\nMetric {best_score:.4f} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    run()
