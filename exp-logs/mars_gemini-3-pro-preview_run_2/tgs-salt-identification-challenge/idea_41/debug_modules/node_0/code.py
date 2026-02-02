import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.data import get_train_val_loaders, get_test_loader, get_pseudo_loader
from library.models import SpecialistTeacher, GeneralistStudent
from library.losses import calc_combined_loss
from library.training import (
    train_model,
    predict_teacher_marginalized,
    generate_submission,
)
from library.utils import unpad_image, rle_decode


def run_demo():
    print("=== Starting Salt Segmentation Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment for Fast Demo...")

    # Override Config for speed
    Config.MAX_SAMPLES = 64  # Small subset of data
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS_TEACHER = 1  # Single epoch
    Config.EPOCHS_STUDENT = 1  # Single epoch
    Config.FOLDS = 1  # Single fold
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.CACHE_DIR = "./working/demo_cache"
    Config.CHECKPOINT_DIR = "./working/demo_checkpoints"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Initialize directories and seeds
    Config.setup()
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Cache Dir: {Config.CACHE_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 2] Testing Data Loaders...")

    # Get Train/Val Loaders
    train_loader, val_loader, depth_stats = get_train_val_loaders()
    test_loader = get_test_loader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Depth Stats: {depth_stats}")

    # Verify Train Batch
    batch = next(iter(train_loader))
    images, masks, depths = batch

    # Assertions for shapes
    # Images: (B, 1, 128, 128) - padded size
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    # Masks: (B, 1, 128, 128)
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect mask shape: {masks.shape}"
    # Depths: (B, 1)
    assert depths.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect depth shape: {depths.shape}"

    print("Data Loader shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 3] Testing Models and Forward Pass...")

    # Instantiate Teacher
    teacher = SpecialistTeacher().to(device)

    # Instantiate Student
    student = GeneralistStudent().to(device)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    # Test Teacher Forward (Expects Image + Depth)
    teacher_logits = teacher(images, depths)
    assert teacher_logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Teacher output shape mismatch"

    # Test Student Forward (Expects Image only)
    # In training mode, returns (logits, depth_pred)
    student.train()
    student_logits, student_depth_pred = student(images)
    assert student_logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Student logits shape mismatch"
    assert student_depth_pred.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Student depth prediction shape mismatch"

    print("Model forward passes successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Testing Loss Functions...")

    # Calculate combined loss for Teacher (Hard targets)
    loss_teacher = calc_combined_loss(teacher_logits, masks, soft_targets=False)
    assert (
        torch.is_tensor(loss_teacher) and loss_teacher.ndim == 0
    ), "Teacher loss should be scalar"
    print(f"Teacher Loss (Initial): {loss_teacher.item():.4f}")

    # Calculate combined loss for Student (Multi-task)
    loss_student = calc_combined_loss(
        student_logits,
        masks,
        pred_depth=student_depth_pred,
        target_depth=depths,
        soft_targets=False,
    )
    assert (
        torch.is_tensor(loss_student) and loss_student.ndim == 0
    ), "Student loss should be scalar"
    print(f"Student Loss (Initial): {loss_student.item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Training Loop (Teacher)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training Teacher (1 Epoch)...")

    optimizer = optim.AdamW(teacher.parameters(), lr=1e-3)
    teacher_save_path = os.path.join(Config.CHECKPOINT_DIR, "demo_teacher.pth")

    trained_teacher = train_model(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        epochs=Config.EPOCHS_TEACHER,
        save_path=teacher_save_path,
        is_teacher=True,
    )

    assert os.path.exists(teacher_save_path), "Teacher checkpoint not saved"
    print("Teacher training complete.")

    # -------------------------------------------------------------------------
    # 6. Marginalized Inference (Pseudo-Labeling)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Marginalized Pseudo-Labels...")

    # We use the trained teacher to predict on test set
    # Using a small subset of depth scans for speed
    scan_depths = [-1.0, 0.0, 1.0]

    soft_masks_dict = predict_teacher_marginalized(
        model=trained_teacher,
        loader=test_loader,
        device=device,
        scan_depths=scan_depths,
    )

    # Verify output
    test_ids = list(soft_masks_dict.keys())
    assert len(test_ids) > 0, "No predictions generated"
    sample_mask = soft_masks_dict[test_ids[0]]

    # Should be (1, 128, 128) or (128, 128) depending on implementation details in predict_teacher_marginalized
    # The function returns pred[0] which is (H, W) if original was (1, H, W)
    # Actually looking at library code: preds_np is (B, 1, H, W), stored as pred[0] -> (1, H, W)
    # Wait, let's check library/training.py:
    # preds_np = avg_preds.cpu().numpy() -> (B, 1, H, W)
    # results[i] = pred[0] -> (H, W)

    assert sample_mask.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Pseudo-label shape mismatch: {sample_mask.shape}"
    assert (
        0.0 <= sample_mask.min() and sample_mask.max() <= 1.0
    ), "Pseudo-labels should be probabilities [0, 1]"

    print(f"Generated pseudo-labels for {len(soft_masks_dict)} test images.")

    # -------------------------------------------------------------------------
    # 7. Training Loop (Student)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Training Student with Pseudo-Labels (1 Epoch)...")

    # Create Pseudo Loader
    pseudo_loader = get_pseudo_loader(soft_masks_dict)

    optimizer_s = optim.AdamW(student.parameters(), lr=1e-3)
    student_save_path = os.path.join(Config.CHECKPOINT_DIR, "demo_student.pth")

    trained_student = train_model(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_s,
        scheduler=None,
        device=device,
        epochs=Config.EPOCHS_STUDENT,
        save_path=student_save_path,
        is_teacher=False,
        unlabeled_loader=pseudo_loader,
    )

    assert os.path.exists(student_save_path), "Student checkpoint not saved"
    print("Student training complete.")

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 8] Generating Submission...")

    submission_file = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    generate_submission(
        model=trained_student,
        test_loader=test_loader,
        val_loader=val_loader,
        device=device,
        output_path=submission_file,
    )

    assert os.path.exists(submission_file), "Submission file not created"

    # Verify content
    df = pd.read_csv(submission_file)
    print(f"Submission rows: {len(df)}")
    assert "id" in df.columns and "rle_mask" in df.columns, "Submission columns missing"

    # Check if we can decode a mask
    if len(df) > 0 and not pd.isna(df.iloc[0]["rle_mask"]):
        rle = df.iloc[0]["rle_mask"]
        decoded = rle_decode(rle)
        assert decoded.shape == (
            Config.ORIG_SIZE,
            Config.ORIG_SIZE,
        ), "Decoded mask shape mismatch"

    print("Submission generated and verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
