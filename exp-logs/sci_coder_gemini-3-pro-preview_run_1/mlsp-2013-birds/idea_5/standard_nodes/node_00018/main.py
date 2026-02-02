import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    create_submission,
)
from library.data import get_dataloaders, create_loader, BirdDataset
from library.model import build_model
from library.engine import train_one_epoch, evaluate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.print_config()

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading (Get raw arrays to manipulate for semi-supervised learning)
    print("\n--- Loading Data ---")
    # We need arrays to combine Train + Pseudo-Labeled Test later
    (train_data, val_data, test_data) = get_dataloaders(
        load_cached_data=True, return_arrays=True
    )

    train_imgs, train_lbls, train_ids = train_data
    val_imgs, val_lbls, val_ids = val_data
    test_imgs, test_lbls, test_ids = test_data

    print(f"Train shape: {train_imgs.shape}")
    print(f"Val shape: {val_imgs.shape}")
    print(f"Test shape: {test_imgs.shape}")

    # Create standard loaders for Stage 1
    train_loader_stage1 = create_loader(
        train_imgs,
        train_lbls,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        transform_mode="train",
    )

    val_loader = create_loader(
        val_imgs,
        val_lbls,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        transform_mode="val",
    )

    # 3. Stage 1: Teacher Training
    print("\n--- Stage 1: Teacher Training (Supervised) ---")
    teacher_model = build_model(device)

    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.TEACHER_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_teacher_auc = 0.0
    best_teacher_state = None

    for epoch in range(Config.TEACHER_EPOCHS):
        avg_loss = train_one_epoch(
            teacher_model, train_loader_stage1, optimizer, criterion, device
        )
        val_loss, val_auc = evaluate(teacher_model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.TEACHER_EPOCHS} - Loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_teacher_auc:
            best_teacher_auc = val_auc
            best_teacher_state = copy.deepcopy(teacher_model.state_dict())
            save_checkpoint(best_teacher_state, True, Config.MODEL_SAVE_PATH_TEACHER)

    print(f"Best Teacher AUC: {best_teacher_auc:.4f}")

    # 4. Stage 2: Pseudo-Labeling
    print("\n--- Stage 2: Pseudo-Labeling Test Data ---")
    # Load best teacher
    teacher_model.load_state_dict(best_teacher_state)

    # Create test loader (no shuffle, val transforms)
    test_loader_pseudo = create_loader(
        test_imgs,
        test_lbls,  # labels are placeholders here
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        transform_mode="val",
    )

    # Predict probabilities
    test_probs = predict(teacher_model, test_loader_pseudo, device)

    # 5. Stage 3: Student Training (Semi-Supervised)
    print("\n--- Stage 3: Student Training (Semi-Supervised) ---")

    # Combine Train (Hard Labels) and Test (Soft Pseudo-Labels)
    # Train labels are 0/1. Test probs are 0.0-1.0.
    # We can treat both as float targets for BCEWithLogitsLoss.

    combined_imgs = np.concatenate([train_imgs, test_imgs], axis=0)
    combined_labels = np.concatenate([train_lbls, test_probs], axis=0)

    print(f"Combined Student Dataset Size: {len(combined_imgs)}")

    # Create Student Loader
    # We use 'train' transforms for the student to regularize
    student_loader = create_loader(
        combined_imgs,
        combined_labels,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        transform_mode="train",
    )

    # Initialize fresh Student Model
    student_model = build_model(device)

    optimizer_student = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_student = CosineAnnealingLR(
        optimizer_student, T_max=Config.STUDENT_EPOCHS
    )

    best_student_auc = 0.0
    best_student_state = None

    for epoch in range(Config.STUDENT_EPOCHS):
        # Note: criterion is still BCEWithLogitsLoss, which handles soft targets fine
        avg_loss = train_one_epoch(
            student_model, student_loader, optimizer_student, criterion, device
        )

        # Validate on original Validation Set
        val_loss, val_auc = evaluate(student_model, val_loader, criterion, device)
        scheduler_student.step()

        print(
            f"Epoch {epoch+1}/{Config.STUDENT_EPOCHS} - Loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_student_auc:
            best_student_auc = val_auc
            best_student_state = copy.deepcopy(student_model.state_dict())
            save_checkpoint(best_student_state, True, Config.MODEL_SAVE_PATH_STUDENT)

    # 6. Final Evaluation & Failure Analysis
    print("\n--- Final Evaluation ---")
    # Load best student
    student_model.load_state_dict(best_student_state)

    # Recalculate metrics on validation set
    final_loss, final_auc = evaluate(student_model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Get predictions
    val_probs = predict(student_model, val_loader, device)

    # Calculate Mean Absolute Error per sample
    # val_lbls is (N, 19), val_probs is (N, 19)
    errors = np.abs(val_probs - val_lbls).mean(axis=1)

    # Calculate a simple feature: Image Brightness (Mean Pixel Value)
    # val_imgs is (N, H, W, 3) uint8
    brightness = val_imgs.mean(axis=(1, 2, 3)) / 255.0

    # Correlation
    if len(errors) > 1:
        correlation = np.corrcoef(errors, brightness)[0, 1]
        print(f"Correlation between Error and Image Brightness: {correlation:.4f}")

        # Identify worst failures
        worst_indices = np.argsort(errors)[-5:][::-1]
        print(f"Top 5 Worst Sample Indices: {val_ids[worst_indices]}")
        print(f"Top 5 Worst Errors: {errors[worst_indices]}")
    else:
        print("Not enough validation samples for correlation analysis.")

    # 7. Submission
    THRESHOLD = 0.9255537489325414
    if final_auc > THRESHOLD:
        print("\n--- Generating Submission ---")
        # Predict on Test Set using Student Model
        # Use the pseudo loader created earlier (same data, val transforms)
        final_test_probs = predict(student_model, test_loader_pseudo, device)

        submission_path = "./submission/submission.csv"
        create_submission(final_test_probs, test_ids, submission_path)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric {final_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
