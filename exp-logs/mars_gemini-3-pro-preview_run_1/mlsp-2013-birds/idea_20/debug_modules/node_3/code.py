import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.models import create_model
from library.engine import fit, inference
from library.distillation import generate_pseudo_labels


def main():
    print("--- Starting Library Demo ---")

    # 1. Setup and Configuration Overrides for Speed
    # We override the Config class attributes to make this run quickly for demonstration.
    print("[1] Configuring environment for rapid demo...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    # Activate SWA immediately for demo purposes
    Config.SWA_START_EPOCH_TEACHER = 0
    Config.SWA_START_EPOCH_STUDENT = 0

    # Reduce ensemble to a single model for this demo to save time
    Config.TEACHER_ARCHS = ["resnet34"]

    # Update paths to a demo-specific directory to avoid overwriting real work
    Config.WORKING_DIR = "./working/demo_run"
    Config.TEACHER_CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.STUDENT_CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PSEUDO_LABELS_PATH = os.path.join(
        Config.WORKING_DIR, "demo_pseudo_labels.parquet"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create directories
    os.makedirs(Config.TEACHER_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.STUDENT_CHECKPOINT_DIR, exist_ok=True)

    print(f"    Working directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    dataloaders = get_dataloaders(pseudo_labels_path=None)
    train_loader = dataloaders["train"]

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]
    rec_ids = batch["rec_id"]

    print(
        f"    Batch Image Shape: {images.shape} (Expected: [{Config.BATCH_SIZE}, 3, {Config.IMG_HEIGHT}, {Config.IMG_WIDTH}])"
    )
    print(
        f"    Batch Label Shape: {labels.shape} (Expected: [{Config.BATCH_SIZE}, {Config.NUM_CLASSES}])"
    )

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), "Image shape mismatch"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Label shape mismatch"
    assert rec_ids.shape[0] == Config.BATCH_SIZE, "Rec ID shape mismatch"

    # 3. Teacher Training Demo
    print("\n[3] Training Teacher Model (ResNet34)...")

    # Create model
    teacher_arch = "resnet34"
    model = create_model(teacher_arch, num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(Config.DEVICE)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Define save path for this specific teacher
    # Note: Config.get_teacher_path returns path based on index and arch
    teacher_save_path = Config.get_teacher_path(0, teacher_arch)

    # Train using the engine
    trained_teacher = fit(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        swa_start_epoch=Config.SWA_START_EPOCH_TEACHER,
        save_path=teacher_save_path,
        patience=None,  # Disable early stopping for this short demo
    )

    assert os.path.exists(
        teacher_save_path
    ), f"Teacher checkpoint not found at {teacher_save_path}"
    print("    Teacher training complete and checkpoint saved.")

    # 4. Pseudo-Label Generation Demo
    print("\n[4] Generating Pseudo-Labels...")

    # The generate_pseudo_labels function relies on Config.TEACHER_ARCHS and loads from disk.
    # We set Config.TEACHER_ARCHS = ["resnet34"] earlier, so it will look for the file we just saved.

    # Force regeneration by ignoring cache if it exists
    if os.path.exists(Config.PSEUDO_LABELS_PATH):
        os.remove(Config.PSEUDO_LABELS_PATH)

    df_pseudo = generate_pseudo_labels(load_cached_data=False)

    print(f"    Pseudo-labels shape: {df_pseudo.shape}")
    print(f"    Columns: {df_pseudo.columns.tolist()[:5]}...")

    # Verify pseudo-labels
    # Test set size is 64 (from metadata analysis)
    # Columns should be rec_id + 19 species
    assert (
        df_pseudo.shape[1] == Config.NUM_CLASSES + 1
    ), "Incorrect number of columns in pseudo-labels"
    assert "rec_id" in df_pseudo.columns, "rec_id column missing"
    assert os.path.exists(
        Config.PSEUDO_LABELS_PATH
    ), "Pseudo-labels parquet file not saved"

    # 5. Student Training Demo
    print("\n[5] Training Student Model with Pseudo-Labels...")

    # Reload dataloaders with pseudo-labels
    # This should combine original train set + pseudo-labeled test set
    dataloaders_student = get_dataloaders(pseudo_labels_path=Config.PSEUDO_LABELS_PATH)

    # Verify dataset size increase
    # Original train (fold 0) is ~208. Test (fold 1) is ~64. Combined should be ~272.
    # Note: The exact numbers depend on the stratification in metadata generation.
    len_train_orig = len(dataloaders["train"].dataset)
    len_train_student = len(dataloaders_student["train"].dataset)

    print(f"    Original Train Size: {len_train_orig}")
    print(f"    Student Train Size: {len_train_student}")

    assert (
        len_train_student > len_train_orig
    ), "Student dataset size did not increase with pseudo-labels"

    # Create Student Model
    student_model = create_model(
        Config.STUDENT_ARCH, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    student_model.to(Config.DEVICE)

    optimizer_student = optim.AdamW(student_model.parameters(), lr=Config.LEARNING_RATE)
    scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=Config.EPOCHS
    )

    student_save_path = Config.get_student_path()

    # Train Student
    trained_student = fit(
        model=student_model,
        train_loader=dataloaders_student["train"],
        val_loader=dataloaders_student["val"],
        optimizer=optimizer_student,
        scheduler=scheduler_student,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        swa_start_epoch=Config.SWA_START_EPOCH_STUDENT,
        save_path=student_save_path,
        patience=None,
    )

    assert os.path.exists(
        student_save_path
    ), f"Student checkpoint not found at {student_save_path}"
    print("    Student training complete.")

    # 6. Inference Demo
    print("\n[6] Running Inference...")

    # Run inference using the trained student model
    # The inference function saves the submission file to Config.SUBMISSION_PATH
    inference(
        model=trained_student,
        test_loader=dataloaders_student["test"],
        device=Config.DEVICE,
        submission_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {df_sub.shape}")
    print(f"    First few rows:\n{df_sub.head()}")

    # Check format: Id, Probability
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"
    # Check number of rows: 64 test samples * 19 classes = 1216 rows
    # Note: sample_submission.csv has 1216 rows + header.
    expected_rows = 64 * 19
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    print("\n--- Library Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
