import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from scipy.stats import pearsonr

# Import from provided library files
from library.utils import set_seed, do_kaggle_metric
from library.dataset import get_loaders
from library.model import ResNet34WideLinkNet
from library.losses import CombinedLoss
from library.engine import (
    train_one_epoch,
    validate,
    generate_soft_targets,
    inference,
    threshold_search,
)

# Configuration
SEED = 42
BATCH_SIZE = 32
TEACHER_EPOCHS = 20
STUDENT_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = "./working"
SUBMISSION_PATH = "./submission/submission.csv"
CACHE_DIR = "./working/idea_17/"


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    print(f"Running on device: {DEVICE}")

    # 2. Stage 1: Teacher Training
    print("\n=== Stage 1: Teacher Training ===")

    # Load Data (Supervised)
    loaders = get_loaders(batch_size=BATCH_SIZE, load_cached_data=True)

    # Initialize Teacher
    teacher = ResNet34WideLinkNet(pretrained=True).to(DEVICE)
    criterion = CombinedLoss().to(DEVICE)
    optimizer = optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS)

    best_teacher_score = 0.0
    best_teacher_path = os.path.join(WORKING_DIR, "teacher_model.pth")

    for epoch in range(TEACHER_EPOCHS):
        train_loss = train_one_epoch(
            teacher, loaders["train"], optimizer, criterion, device=DEVICE
        )
        val_loss, val_score, _, _ = validate(
            teacher, loaders["val"], criterion, device=DEVICE
        )
        scheduler.step()

        # Save best teacher
        if val_score > best_teacher_score:
            best_teacher_score = val_score
            torch.save(teacher.state_dict(), best_teacher_path)

    print(f"Stage 1 Complete. Best Teacher mAP: {best_teacher_score:.4f}")

    # 3. Stage 2: Student Distillation
    print("\n=== Stage 2: Student Distillation ===")

    # Load Best Teacher
    teacher.load_state_dict(torch.load(best_teacher_path, map_location=DEVICE))

    # Generate Soft Targets
    print("Generating soft targets for test set...")
    soft_targets = generate_soft_targets(teacher, loaders["test"], device=DEVICE)

    # Reload Data (Semi-Supervised)
    # Note: get_loaders handles mixing train and test data when soft_test_masks is provided
    loaders_student = get_loaders(
        batch_size=BATCH_SIZE, load_cached_data=True, soft_test_masks=soft_targets
    )

    # Initialize Student
    student = ResNet34WideLinkNet(pretrained=True).to(DEVICE)
    optimizer_student = optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=STUDENT_EPOCHS
    )

    best_student_score = 0.0
    best_student_path = os.path.join(WORKING_DIR, "student_model.pth")

    # We validate on the original validation set
    val_loader = loaders["val"]

    for epoch in range(STUDENT_EPOCHS):
        train_loss = train_one_epoch(
            student,
            loaders_student["train"],
            optimizer_student,
            criterion,
            device=DEVICE,
        )
        val_loss, val_score, _, _ = validate(
            student, val_loader, criterion, device=DEVICE
        )
        scheduler_student.step()

        if val_score > best_student_score:
            best_student_score = val_score
            torch.save(student.state_dict(), best_student_path)

    print(f"Stage 2 Complete. Best Student mAP: {best_student_score:.4f}")

    # 4. Final Validation & Analysis
    print("\n=== Final Evaluation & Analysis ===")

    # Load best student
    student.load_state_dict(torch.load(best_student_path, map_location=DEVICE))

    # Get predictions
    val_loss, final_map, val_preds, val_targets = validate(
        student, val_loader, criterion, device=DEVICE
    )

    # REQUIRED: Print Final Metric
    print(f"Final Validation Metric: {final_map}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate IoU per image (at threshold 0.5 for analysis)
    preds_bin = (val_preds > 0.5).astype(np.uint8)
    targets_bin = val_targets.astype(np.uint8)

    ious = []
    for i in range(len(preds_bin)):
        intersection = np.sum(preds_bin[i] & targets_bin[i])
        union = np.sum(preds_bin[i] | targets_bin[i])
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)

    ious = np.array(ious)
    errors = 1.0 - ious

    # Load validation depths from cache
    # The cache files are created by prepare_data in library/dataset.py
    val_depths = np.load(os.path.join(CACHE_DIR, "val_depths.npy"))

    # Ensure lengths match
    if len(errors) == len(val_depths):
        corr, _ = pearsonr(val_depths, errors)
        print(f"Correlation between Error (1-IoU) and Depth: {corr:.4f}")
    else:
        print(
            f"Warning: Length mismatch for analysis. Errors: {len(errors)}, Depths: {len(val_depths)}"
        )

    # 5. Submission
    if final_map > 0.7985:
        print("\n=== Generating Submission ===")

        # Optimize Threshold
        best_threshold, best_thresh_score = threshold_search(val_preds, val_targets)
        print(
            f"Optimal Threshold: {best_threshold:.4f} (Score: {best_thresh_score:.4f})"
        )

        # Inference
        inference(
            student,
            loaders["test"],
            threshold=best_threshold,
            output_path=SUBMISSION_PATH,
            device=DEVICE,
        )
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score {final_map:.4f} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
