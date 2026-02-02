import os
import sys
import numpy as np
import torch
import pandas as pd
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import rle_encode, rle_decode, do_kaggle_metric
from library.models import SpecialistTeacher, GeneralistStudent
from library.losses import MixedLoss, StudentLoss
from library.data import get_stage1_loaders, get_test_loader
from library.pipeline import Pipeline


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # =========================================================================
    # 1. Configuration Setup for Demo
    # =========================================================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config parameters to run fast
    Config.CACHE_DIR = "./working/demo_execution/cache"
    Config.TEACHER_CHECKPOINT_DIR = "./working/demo_execution/teacher_ckpt"
    Config.STUDENT_CHECKPOINT_DIR = "./working/demo_execution/student_ckpt"
    Config.SUBMISSION_FILE = "./working/demo_execution/submission.csv"

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.TEACHER_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.STUDENT_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Set hyperparams for speed
    Config.SEED = 42
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.STAGE1_EPOCHS = 1
    Config.STAGE3_EPOCHS = 1
    Config.STAGE1_FOLDS = 2  # Only run 2 folds max (we will likely just run one)
    Config.MAX_TRAIN_SAMPLES = 50  # Limit dataset size significantly
    Config.MAX_VAL_SAMPLES = 20

    # Setup environment (seeds)
    Config.setup()

    print("Configuration updated: 1 Epoch, Batch Size 8, Max Samples 50.")

    # =========================================================================
    # 2. Verify Utilities
    # =========================================================================
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a square of salt

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(dummy_mask, decoded), "RLE decode did not match original mask"
    print(" - RLE Encode/Decode: OK")

    # Test Metric
    # Perfect match
    score_perfect = do_kaggle_metric(dummy_mask, dummy_mask)
    assert np.isclose(score_perfect, 1.0), "Metric should be 1.0 for perfect match"

    # No overlap
    dummy_pred_wrong = np.zeros_like(dummy_mask)
    dummy_pred_wrong[50:60, 50:60] = 1
    score_zero = do_kaggle_metric(dummy_pred_wrong, dummy_mask)
    assert np.isclose(score_zero, 0.0), "Metric should be 0.0 for no overlap"
    print(" - Kaggle Metric Calculation: OK")

    # =========================================================================
    # 3. Verify Models
    # =========================================================================
    print("\n[3] Verifying Models (Forward Pass)...")

    device = Config.DEVICE
    B, C, H, W = 4, 1, 128, 128

    # Dummy inputs
    x = torch.randn(B, C, H, W).to(device)
    depth = torch.randn(B, 1).to(device)

    # Test Specialist Teacher
    teacher = SpecialistTeacher().to(device)
    teacher_out = teacher(x, depth)

    assert teacher_out.shape == (
        B,
        1,
        H,
        W,
    ), f"Teacher output shape mismatch: {teacher_out.shape}"
    print(" - SpecialistTeacher Forward: OK")

    # Test Generalist Student
    student = GeneralistStudent().to(device)
    student_mask, student_depth = student(x)

    assert student_mask.shape == (
        B,
        1,
        H,
        W,
    ), f"Student mask output mismatch: {student_mask.shape}"
    assert student_depth.shape == (
        B,
        1,
    ), f"Student depth output mismatch: {student_depth.shape}"
    print(" - GeneralistStudent Forward: OK")

    # =========================================================================
    # 4. Verify Losses
    # =========================================================================
    print("\n[4] Verifying Loss Functions...")

    # Mixed Loss (BCE + Lovasz)
    mixed_crit = MixedLoss()
    # Create targets (0 or 1)
    targets = (torch.rand(B, 1, H, W) > 0.5).float().to(device)
    loss_val = mixed_crit(teacher_out, targets)

    assert not torch.isnan(loss_val), "MixedLoss returned NaN"
    assert loss_val.item() >= 0, "MixedLoss should be non-negative"
    print(" - MixedLoss: OK")

    # Student Loss
    student_crit = StudentLoss()
    # Case 1: Labeled (with depth target)
    loss_s_labeled = student_crit(student_mask, student_depth, targets, depth)
    assert not torch.isnan(loss_s_labeled), "StudentLoss (Labeled) returned NaN"

    # Case 2: Unlabeled (soft targets, no depth target)
    soft_targets = torch.rand(B, 1, H, W).to(device)
    loss_s_unlabeled = student_crit(
        student_mask, student_depth, soft_targets, depth_targets=None
    )
    assert not torch.isnan(loss_s_unlabeled), "StudentLoss (Unlabeled) returned NaN"
    print(" - StudentLoss: OK")

    # =========================================================================
    # 5. Verify Data Loading
    # =========================================================================
    print("\n[5] Verifying Data Loading...")

    # This will trigger processing and caching (might take a few seconds)
    train_loader, val_loader = get_stage1_loaders(
        fold=0, load_cached_data=False, debug=True
    )

    batch = next(iter(train_loader))
    img_batch = batch["image"]
    mask_batch = batch["mask"]
    depth_batch = batch["depth"]
    id_batch = batch["id"]

    assert img_batch.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Image batch shape wrong: {img_batch.shape}"
    assert mask_batch.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Mask batch shape wrong: {mask_batch.shape}"
    assert depth_batch.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Depth batch shape wrong: {depth_batch.shape}"
    assert len(id_batch) == Config.BATCH_SIZE, "ID batch length wrong"
    print(" - Data Loaders: OK")

    # =========================================================================
    # 6. Pipeline Execution Simulation
    # =========================================================================
    print("\n[6] Simulating Pipeline Execution...")

    pipeline = Pipeline()

    # A. Train Teacher (Fold 0)
    print(" -> Training Teacher (Fold 0)...")
    best_map, teacher_path = pipeline.train_teacher_fold(
        fold=0, epochs=Config.STAGE1_EPOCHS
    )
    assert os.path.exists(teacher_path), "Teacher checkpoint not saved"
    print(f"    Teacher trained. Best mAP: {best_map:.4f}")

    # B. Generate Pseudo Labels
    # We create a dummy list of teacher paths (using the one we just trained)
    print(" -> Generating Pseudo Labels...")
    # For demo, we just use the one model we trained
    teacher_paths = [teacher_path]

    # To save time, we will manually subset the test loader inside the pipeline call?
    # No, we can't easily modify the pipeline method internals, but we can rely on Config.MAX_TRAIN_SAMPLES
    # affecting the test loader if we force it.
    # Actually, get_test_loader reads Config.TEST_METADATA_PATH.
    # We will let it run; with batch size 8 and limited compute, it might process 1000 images.
    # To speed it up for demo, we can hack the test metadata loading or just let it run (1000 images inference is fast on GPU).
    # Let's create a temporary smaller test metadata file for speed.

    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    temp_test_csv = os.path.join(Config.WORKING_DIR, "temp_test_subset.csv")
    full_test_df.head(10).to_csv(temp_test_csv, index=False)

    # Temporarily point Config to this subset
    original_test_path = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = temp_test_csv
    # Clear cache to force reload of test data
    if os.path.exists(os.path.join(Config.CACHE_DIR, "test_images.npy")):
        os.remove(os.path.join(Config.CACHE_DIR, "test_images.npy"))

    pseudo_labels = pipeline.generate_marginalized_labels(teacher_paths)
    assert len(pseudo_labels) > 0, "No pseudo labels generated"
    print(f"    Pseudo labels generated for {len(pseudo_labels)} images.")

    # C. Train Student
    print(" -> Training Student...")
    student_path = pipeline.train_student_distillation(
        pseudo_labels, epochs=Config.STAGE3_EPOCHS
    )
    assert os.path.exists(student_path), "Student checkpoint not saved"
    print("    Student trained.")

    # D. Optimize Threshold
    print(" -> Optimizing Threshold...")
    best_thresh = pipeline.optimize_threshold(student_path)
    assert 0.0 < best_thresh < 1.0, f"Invalid threshold: {best_thresh}"
    print(f"    Optimal Threshold: {best_thresh}")

    # E. Generate Submission
    print(" -> Generating Submission...")
    pipeline.generate_submission(student_path, best_thresh)
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission generated with {len(sub_df)} rows.")
    assert len(sub_df) == 10, "Submission should have 10 rows (based on our subset)"

    # Restore Config
    Config.TEST_METADATA_PATH = original_test_path

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
