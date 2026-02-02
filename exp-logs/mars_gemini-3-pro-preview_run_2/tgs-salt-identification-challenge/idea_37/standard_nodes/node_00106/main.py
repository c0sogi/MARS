import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import (
    DEVICE,
    CACHE_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    STAGE1_EPOCHS,
    STAGE3_EPOCHS,
    SEED,
    IMG_HEIGHT,
    IMG_WIDTH,
)
from library.utils import set_seed, calc_map, calc_iou_batch
from library.dataset import SaltDataset, get_transforms
from library.models import SaltNet
from library.engine import SaltEngine

# Override epochs for fast baseline execution within time limit
# 50 epochs is too long for < 1 hour. We use a smaller number that allows convergence.
FAST_EPOCHS_TEACHER = 8
FAST_EPOCHS_STUDENT = 8


def run_stage1_teacher():
    """
    Stage 1: Train the Specialist Teacher (Depth-Injected) on labeled data.
    """
    print("\n=== Stage 1: Specialist Teacher Training ===")

    # Initialize Model
    model = SaltNet(mode="teacher")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FAST_EPOCHS_TEACHER
    )

    # Data Loaders (Fold 0 equivalent: Train on train.csv, Val on val.csv)
    train_dataset = SaltDataset(mode="train", transform=get_transforms("train"))
    val_dataset = SaltDataset(mode="val", transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Engine
    engine = SaltEngine(
        model, device=DEVICE, optimizer=optimizer, scheduler=scheduler, mode="teacher"
    )

    best_map = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "teacher_best.pth")

    for epoch in range(FAST_EPOCHS_TEACHER):
        loss = engine.train_one_epoch(train_loader, epoch + 1)
        val_map, _ = engine.validate(val_loader)

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"  New Best Teacher mAP: {best_map:.4f}")

    print(f"Stage 1 Complete. Best Teacher mAP: {best_map:.4f}")
    return best_model_path


def run_stage2_pseudolabels(teacher_path):
    """
    Stage 2: Generate Marginalized Soft Pseudo-Labels for the Test Set.
    """
    print("\n=== Stage 2: Marginalized Pseudo-Label Generation ===")

    # Load Teacher
    model = SaltNet(mode="teacher")
    model.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Test Dataset (No transforms needed for generation, just normalization)
    test_dataset = SaltDataset(mode="test", transform=get_transforms("val"))

    # We iterate manually to use predict_marginalized
    pseudo_labels = {}

    print(f"Generating pseudo-labels for {len(test_dataset)} test images...")

    # Process one by one (or could batch, but predict_marginalized is usually single-image or small batch)
    # Given the complexity of marginalization, simple loop is safer for memory.
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            img = sample["image"].unsqueeze(0)  # Add batch dim
            img_id = sample["id"]

            # Marginalized Inference (Depth Scan)
            # Returns (1, 1, H, W) soft probs
            soft_mask = SaltEngine.predict_marginalized(model, img, DEVICE)

            # Squeeze to (H, W) numpy
            soft_mask_np = soft_mask.squeeze().cpu().numpy()

            pseudo_labels[img_id] = soft_mask_np

    print("Stage 2 Complete. Pseudo-labels generated.")
    return pseudo_labels


def run_stage3_student(pseudo_labels):
    """
    Stage 3: Train Generalist Student on Combined Data (Distillation).
    """
    print("\n=== Stage 3: Student Distillation ===")

    # Initialize Student Model
    model = SaltNet(mode="student")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FAST_EPOCHS_STUDENT
    )

    # Combined Dataset (Train + Test with Pseudo-labels)
    # mode="semi_supervised" loads train+val+test metadata
    # pseudo_labels dict provides masks for test IDs
    combined_dataset = SaltDataset(
        mode="semi_supervised",
        pseudo_labels=pseudo_labels,
        transform=get_transforms("train"),
    )

    # Validation Set (Same as Stage 1)
    val_dataset = SaltDataset(mode="val", transform=get_transforms("val"))

    train_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Engine
    engine = SaltEngine(
        model, device=DEVICE, optimizer=optimizer, scheduler=scheduler, mode="student"
    )

    best_map = 0.0
    best_thresh = 0.5
    best_model_path = os.path.join(CHECKPOINT_DIR, "student_best.pth")

    for epoch in range(FAST_EPOCHS_STUDENT):
        loss = engine.train_one_epoch(train_loader, epoch + 1)
        val_map, thresh = engine.validate(val_loader)

        if val_map > best_map:
            best_map = val_map
            best_thresh = thresh
            torch.save(model.state_dict(), best_model_path)
            print(f"  New Best Student mAP: {best_map:.4f}")

    print(f"Stage 3 Complete. Best Student mAP: {best_map:.4f}")
    return best_model_path, best_thresh


def failure_analysis(model, val_loader, device):
    """
    Analyzes model performance relative to depth.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    ious = []
    depths = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            batch_depths = batch["depth"].cpu().numpy().flatten()

            # Student inference
            logits, _ = model(images)
            preds = (torch.sigmoid(logits) > 0.5).byte()

            # Calculate IoU per image
            masks_uint8 = (masks > 0.5).byte()
            batch_ious = calc_iou_batch(preds.cpu().numpy(), masks_uint8.cpu().numpy())

            ious.extend(batch_ious)
            depths.extend(batch_depths)

    ious = np.array(ious)
    depths = np.array(depths)
    errors = 1.0 - ious

    # Correlation
    corr = np.corrcoef(errors, depths)[0, 1]
    print(f"Correlation between Error (1-IoU) and Depth: {corr:.4f}")

    if abs(corr) > 0.1:
        print(
            "  -> Significant correlation detected. Depth paradox partially mitigated but still present."
        )
    else:
        print(
            "  -> Low correlation. Student has successfully generalized across depths."
        )


def main():
    set_seed(SEED)

    # 1. Train Teacher
    teacher_path = run_stage1_teacher()

    # 2. Generate Pseudo-Labels
    pseudo_labels = run_stage2_pseudolabels(teacher_path)

    # 3. Train Student
    student_path, best_thresh = run_stage3_student(pseudo_labels)

    # 4. Final Validation & Analysis
    print("\n=== Final Evaluation ===")

    # Load best student
    student_model = SaltNet(mode="student")
    student_model.load_state_dict(torch.load(student_path, map_location=DEVICE))
    student_model.to(DEVICE)
    student_model.eval()

    val_dataset = SaltDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Engine for validation
    engine = SaltEngine(student_model, device=DEVICE, mode="student")

    # Compute Final Metric
    final_map, _ = engine.validate(val_loader)
    print(f"Final Validation Metric: {final_map:.10f}")

    # Failure Analysis
    failure_analysis(student_model, val_loader, DEVICE)

    # 5. Submission
    # Threshold condition as per task description
    if final_map > 0.7985:
        print("\nValidation metric meets threshold. Generating submission...")

        # Test Loader for submission
        test_dataset = SaltDataset(mode="test", transform=get_transforms("val"))
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        engine.generate_submission_csv(test_loader, threshold=best_thresh)
    else:
        print(
            f"\nValidation metric {final_map:.4f} is below threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
