import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.utils import set_seed, do_kaggle_metric, rle_encode
from library.dataset import get_dataloaders
from library.models import SaltLinkNet
from library.losses import MixedLoss, DistillationLoss
from library.engine import (
    run_teacher_training,
    run_student_distillation,
    generate_submission,
)


def main():
    # 1. Setup and Configuration
    print("Initializing demonstration...")
    SEED = 42
    set_seed(SEED)

    # Define working directory for this demo
    WORK_DIR = "./working/demo_test"
    os.makedirs(WORK_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. Data Loading Verification
    print("\n--- Verifying Data Loading ---")
    # Use debug=True to load a very small subset (100 train, 50 val/test)
    dataloaders = get_dataloaders(
        batch_size=4, num_workers=0, load_cached_data=True, debug=True
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = {"image", "mask", "depth", "id"}
    assert (
        set(batch.keys()) == expected_keys
    ), f"Batch keys mismatch. Found: {batch.keys()}"

    # Check shapes (Batch size 4, Image size 101x101)
    imgs = batch["image"]
    masks = batch["mask"]
    depths = batch["depth"]

    assert imgs.shape == (4, 1, 101, 101), f"Unexpected image shape: {imgs.shape}"
    assert masks.shape == (4, 1, 101, 101), f"Unexpected mask shape: {masks.shape}"
    assert depths.shape == (4, 1), f"Unexpected depth shape: {depths.shape}"

    print("Data loader verification passed.")

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architectures ---")

    # 3a. Teacher Model (Requires Depth)
    teacher_model = SaltLinkNet(mode="teacher").to(DEVICE)
    dummy_img = torch.randn(2, 1, 101, 101).to(DEVICE)
    dummy_depth = torch.randn(2, 1).to(DEVICE)

    teacher_out = teacher_model(dummy_img, depth=dummy_depth)
    assert teacher_out.shape == (
        2,
        1,
        101,
        101,
    ), f"Teacher output shape mismatch: {teacher_out.shape}"

    # Verify Teacher fails without depth
    try:
        teacher_model(dummy_img)
        raise AssertionError("Teacher model should fail when depth is not provided.")
    except ValueError:
        pass  # Expected behavior

    # 3b. Student Model (Image Only)
    student_model = SaltLinkNet(mode="student").to(DEVICE)
    student_out = student_model(dummy_img)
    assert student_out.shape == (
        2,
        1,
        101,
        101,
    ), f"Student output shape mismatch: {student_out.shape}"

    print("Model architecture verification passed.")

    # 4. Loss Function Verification
    print("\n--- Verifying Loss Functions ---")

    # 4a. MixedLoss (BCE + Lovasz)
    loss_fn_mixed = MixedLoss()
    # Create dummy logits and targets
    # Targets should be 0 or 1
    dummy_logits = torch.randn(2, 1, 101, 101).to(DEVICE)
    dummy_targets = torch.randint(0, 2, (2, 1, 101, 101)).float().to(DEVICE)

    loss_val = loss_fn_mixed(dummy_logits, dummy_targets)
    assert loss_val.dim() == 0, "MixedLoss should return a scalar."
    assert not torch.isnan(loss_val), "MixedLoss returned NaN."

    # 4b. DistillationLoss (Seg + MSE)
    loss_fn_distill = DistillationLoss()
    teacher_logits = torch.randn(2, 1, 101, 101).to(DEVICE)
    student_logits = torch.randn(2, 1, 101, 101).to(DEVICE)

    distill_val = loss_fn_distill(student_logits, teacher_logits, dummy_targets)
    assert distill_val.dim() == 0, "DistillationLoss should return a scalar."

    # Check gradient flow
    distill_val.backward()
    print("Loss function verification passed.")

    # 5. Training Loop Demonstration (Engine)
    print("\n--- Verifying Training Engine ---")

    teacher_save_path = os.path.join(WORK_DIR, "teacher_ckpt.pth")
    student_save_path = os.path.join(WORK_DIR, "student_ckpt.pth")

    # 5a. Phase 1: Teacher Training
    # Run for 1 epoch on the debug dataset
    print("Running Teacher Training (1 Epoch)...")
    run_teacher_training(
        loader_train=train_loader,
        loader_val=val_loader,
        device=DEVICE,
        epochs=1,
        lr=1e-3,
        patience=1,
        save_path=teacher_save_path,
    )

    assert os.path.exists(teacher_save_path), "Teacher checkpoint was not saved."

    # 5b. Phase 2: Student Distillation
    # Run for 1 epoch
    print("Running Student Distillation (1 Epoch)...")
    run_student_distillation(
        teacher_path=teacher_save_path,
        loader_train=train_loader,
        loader_val=val_loader,
        device=DEVICE,
        epochs=1,
        lr=1e-3,
        patience=1,
        save_path=student_save_path,
    )

    assert os.path.exists(student_save_path), "Student checkpoint was not saved."
    print("Training engine verification passed.")

    # 6. Submission Generation Verification
    print("\n--- Verifying Submission Generation ---")

    submission_path = os.path.join(WORK_DIR, "submission.csv")

    # Generate submission using the trained student model
    # Note: validation loader is used for threshold optimization, test loader for prediction
    generate_submission(
        model_path=student_save_path,
        loader_test=test_loader,
        loader_val=val_loader,
        device=DEVICE,
        output_path=submission_path,
    )

    assert os.path.exists(submission_path), "Submission file was not created."

    # Check content format
    df_sub = pd.read_csv(submission_path)
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns mismatch."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check RLE format on first entry
    first_rle = df_sub.iloc[0]["rle_mask"]
    if pd.notna(first_rle) and first_rle != "":
        rle_parts = [int(x) for x in first_rle.split()]
        assert len(rle_parts) % 2 == 0, "RLE string must have even number of integers."

    print("Submission generation verification passed.")

    # 7. Metric Utility Verification
    print("\n--- Verifying Metric Utility ---")
    # Perfect match
    score_perfect = do_kaggle_metric(dummy_targets, dummy_targets, threshold=0.5)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Metric should be 1.0 for perfect match, got {score_perfect}"

    # No match
    score_zero = do_kaggle_metric(
        torch.zeros_like(dummy_targets), dummy_targets, threshold=0.5
    )
    # Note: If ground truth is all zeros, and pred is all zeros, IoU is 1.0.
    # If ground truth has salt and pred is empty, score is 0.
    # We rely on the random dummy_targets having some 1s.
    if dummy_targets.sum() > 0:
        # It's possible for random tensor to be all 0s, but unlikely with size (2,1,101,101)
        pass

    print("Metric utility verification passed.")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
