import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import (
    pad_image,
    unpad_image,
    rle_encode,
    rle_decode,
)
from library.dataset import get_loaders
from library.models import TeacherLinkNet, StudentLinkNet
from library.losses import StudentLoss
from library.pipeline import (
    run_stage1_teacher,
    run_stage2_student_ensemble,
    run_stage3_self_training,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demo Script...")

    # =========================================================================
    # 1. Configuration Setup for Rapid Execution
    # =========================================================================
    print("Configuring for DEBUG mode (small dataset, minimal epochs)...")

    # Modify Config singleton to run a fast demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Use a tiny subset of data
    Config.BATCH_SIZE = 4  # Small batch size

    # Set epochs to 1 (Note: pipeline functions might override this to 2 in debug mode, which is fine)
    Config.NUM_EPOCHS_TEACHER = 1
    Config.NUM_EPOCHS_STUDENT = 1
    Config.NUM_EPOCHS_FINAL = 1

    # Use a separate cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_execution/cache"
    Config.SUBMISSION_PATH = "./working/demo_execution/submission.csv"

    # Ensure clean state
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print("\n[1/5] Verifying Utility Functions...")

    # Test Padding Logic
    orig_shape = (101, 101)
    dummy_img = np.random.randint(0, 255, orig_shape, dtype=np.uint8)
    padded = pad_image(dummy_img)

    assert padded.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Padding failed. Expected ({Config.IMG_SIZE}, {Config.IMG_SIZE}), got {padded.shape}"

    unpadded = unpad_image(padded)
    assert (
        unpadded.shape == orig_shape
    ), f"Unpadding failed. Expected {orig_shape}, got {unpadded.shape}"

    # Test RLE Encoding/Decoding
    # Create a simple mask
    mask_orig = np.zeros(orig_shape, dtype=np.uint8)
    mask_orig[10:20, 10:20] = 1

    rle_str = rle_encode(mask_orig)
    mask_decoded = rle_decode(rle_str, orig_shape)

    assert np.array_equal(mask_orig, mask_decoded), "RLE Encode -> Decode cycle failed."
    print("Utils verification passed.")

    # =========================================================================
    # 3. Verify Data Loading and Processing
    # =========================================================================
    print("\n[2/5] Verifying Data Loading...")

    # This will trigger process_data, caching the debug subset
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch a single batch
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    depths = batch["depth"]
    ids = batch["id"]

    print(
        f"Batch loaded. Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE)
    assert masks.shape == (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE)
    assert depths.shape == (Config.BATCH_SIZE, 1)
    print("Data loading verification passed.")

    # =========================================================================
    # 4. Verify Models and Loss Functions
    # =========================================================================
    print("\n[3/5] Verifying Models & Loss...")
    device = Config.DEVICE

    # Move batch to device
    images = images.to(device).float()
    masks = masks.to(device).float()
    depths = depths.to(device).float()

    # --- Teacher Model (Inputs: Image + Depth) ---
    teacher = TeacherLinkNet(num_classes=1).to(device)
    teacher_logits = teacher(images, depths)

    assert teacher_logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Teacher output shape mismatch"

    # --- Student Model (Inputs: Image only) ---
    student = StudentLinkNet(num_classes=1).to(device)
    student_out = student(images)

    assert (
        "logits" in student_out and "depth" in student_out
    ), "Student output dict keys missing"
    assert student_out["logits"].shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Student logits shape mismatch"
    assert student_out["depth"].shape == (
        Config.BATCH_SIZE,
        1,
    ), "Student depth prediction shape mismatch"

    # --- Loss Function ---
    loss_fn = StudentLoss()
    # Calculate loss with distillation (using teacher logits)
    loss, metrics = loss_fn(student_out, masks, depths, teacher_logits=teacher_logits)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"Computed Loss: {loss.item():.4f}")
    print("Model and Loss verification passed.")

    # =========================================================================
    # 5. Execute Pipeline Stages
    # =========================================================================
    print("\n[4/5] Executing Pipeline Stages...")

    # --- Stage 1: Train Teacher ---
    # debug=True reduces epochs and prints less
    print("Running Stage 1: Teacher Training...")
    teacher_ckpt_path = run_stage1_teacher(debug=True)

    assert os.path.exists(
        teacher_ckpt_path
    ), f"Stage 1 failed: {teacher_ckpt_path} not found."

    # --- Stage 2: Train Student Ensemble (Distillation) ---
    print("Running Stage 2: Student Ensemble Training...")
    # This will run K-Fold (limited to 1 fold in debug mode)
    student_ckpt_paths = run_stage2_student_ensemble(teacher_ckpt_path, debug=True)

    assert len(student_ckpt_paths) > 0, "Stage 2 failed: No student models saved."
    assert os.path.exists(student_ckpt_paths[0]), "Stage 2 model file missing."

    # --- Stage 3: Self-Training (Noisy Student) ---
    print("Running Stage 3: Self-Training...")
    # This generates pseudo-labels and retrains
    run_stage3_self_training(student_ckpt_paths, debug=True)

    # =========================================================================
    # 6. Verify Submission
    # =========================================================================
    print("\n[5/5] Verifying Submission...")

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        print(f"Rows: {len(df_sub)}")
        print(df_sub.head())

        # Basic checks
        assert (
            "id" in df_sub.columns and "rle_mask" in df_sub.columns
        ), "Submission columns missing"
        assert len(df_sub) > 0, "Submission file is empty"

        # Check if RLE strings are valid (or empty)
        sample_rle = df_sub.iloc[0]["rle_mask"]
        if pd.notna(sample_rle) and sample_rle != "":
            # Try decoding
            try:
                rle_decode(sample_rle)
            except Exception as e:
                raise AssertionError(f"Failed to decode submission RLE: {e}")
    else:
        raise FileNotFoundError("Submission file was not generated by Stage 3.")

    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    main()
