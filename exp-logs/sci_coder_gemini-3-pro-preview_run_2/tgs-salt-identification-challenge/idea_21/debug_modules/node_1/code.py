import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import provided library modules
from library.config import Config, seed_everything
from library.utils import (
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
    calc_iou,
    calc_map_score,
)
from library.dataset import SaltDataset, get_dataloaders, get_test_loader
from library.models import PrivilegedTeacher, MultiTaskStudent
from library.losses import SegmentationLoss, DistillationLoss
from library.trainer import SaltTrainer


def setup_demo_config():
    """
    Overrides Config parameters for a quick demonstration run.
    Creates a subset of metadata files.
    """
    print("--- Setting up Demonstration Configuration ---")

    # Define demo paths
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Create subset CSVs
    # Read original metadata
    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Sample small subsets (enough for a few batches)
    train_subset = train_full.head(16).copy()
    val_subset = val_full.head(8).copy()
    test_subset = test_full.head(8).copy()

    # Save subsets
    train_csv_path = os.path.join(demo_dir, "train_subset.csv")
    val_csv_path = os.path.join(demo_dir, "val_subset.csv")
    test_csv_path = os.path.join(demo_dir, "test_subset.csv")

    train_subset.to_csv(train_csv_path, index=False)
    val_subset.to_csv(val_csv_path, index=False)
    test_subset.to_csv(test_csv_path, index=False)

    # Override Config attributes
    Config.TRAIN_CSV = train_csv_path
    Config.VAL_CSV = val_csv_path
    Config.TEST_CSV = test_csv_path
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.TEACHER_CHECKPOINT = os.path.join(
        demo_dir, "checkpoints", "teacher_best.pth"
    )
    Config.STUDENT_CHECKPOINT = os.path.join(
        demo_dir, "checkpoints", "student_best.pth"
    )

    # Reduce compute load
    Config.BATCH_SIZE = 4
    Config.TEACHER_EPOCHS = 1
    Config.STUDENT_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure checkpoint dir exists
    os.makedirs(os.path.dirname(Config.TEACHER_CHECKPOINT), exist_ok=True)

    # Clean cache to force reload
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    print(f"Demo configuration applied. Working dir: {Config.WORKING_DIR}")


def verify_utilities():
    """
    Verifies the correctness of utility functions.
    """
    print("\n--- Verifying Utilities ---")

    # 1. RLE Encoding/Decoding
    # Create a dummy mask (101x101) with a known pattern
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # A 10x10 square

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    if not np.array_equal(mask, decoded):
        raise AssertionError("RLE Encode -> Decode cycle failed.")
    print("RLE Encode/Decode: OK")

    # 2. Padding/Unpadding
    padded = pad_image(mask, target_size=128)
    if padded.shape != (128, 128):
        raise AssertionError(
            f"Pad image shape mismatch. Expected (128, 128), got {padded.shape}"
        )

    unpadded = unpad_image(padded, original_shape=(101, 101))
    if not np.array_equal(mask, unpadded):
        raise AssertionError("Pad -> Unpad cycle failed. Content mismatch.")
    print("Pad/Unpad: OK")

    # 3. IoU Calculation
    iou_score = calc_iou(mask, mask)
    if not np.isclose(iou_score, 1.0):
        raise AssertionError(f"IoU for identical masks should be 1.0, got {iou_score}")

    empty_mask = np.zeros_like(mask)
    iou_empty = calc_iou(mask, empty_mask)
    if not np.isclose(iou_empty, 0.0):
        raise AssertionError(f"IoU for disjoint masks should be 0.0, got {iou_empty}")
    print("IoU Calculation: OK")


def verify_models(device):
    """
    Verifies model instantiation and forward passes.
    """
    print("\n--- Verifying Models ---")

    batch_size = 2
    dummy_img = torch.randn(batch_size, 1, 128, 128).to(device)
    dummy_depth = torch.randn(batch_size).to(device)

    # 1. Teacher Model
    teacher = PrivilegedTeacher().to(device)
    try:
        t_out = teacher(dummy_img, dummy_depth)
        if t_out.shape != (batch_size, 1, 128, 128):
            raise AssertionError(f"Teacher output shape mismatch: {t_out.shape}")
        print("PrivilegedTeacher Forward Pass: OK")
    except Exception as e:
        raise RuntimeError(f"PrivilegedTeacher failed: {e}")

    # 2. Student Model
    student = MultiTaskStudent().to(device)
    try:
        s_logits, s_depth = student(dummy_img)
        if s_logits.shape != (batch_size, 1, 128, 128):
            raise AssertionError(f"Student logits shape mismatch: {s_logits.shape}")
        if s_depth.shape[0] != batch_size:
            raise AssertionError(f"Student depth shape mismatch: {s_depth.shape}")
        print("MultiTaskStudent Forward Pass: OK")
    except Exception as e:
        raise RuntimeError(f"MultiTaskStudent failed: {e}")

    return teacher, student


def run_training_pipeline(device):
    """
    Demonstrates the two-phase training process.
    """
    print("\n--- Running Training Pipeline Demo ---")

    # 1. Load Data
    # load_cached_data=False ensures we read the subset CSVs created in setup
    print("Loading datasets...")
    train_loader, val_loader, depth_stats = get_dataloaders(load_cached_data=False)

    trainer = SaltTrainer(device=device)

    # 2. Train Teacher (Phase 1)
    print("Phase 1: Training Privileged Teacher...")
    teacher = PrivilegedTeacher().to(device)
    teacher = trainer.fit_teacher(
        teacher, train_loader, val_loader, epochs=Config.TEACHER_EPOCHS
    )

    # 3. Train Student (Phase 2)
    print("Phase 2: Distilling to Student...")
    student = MultiTaskStudent().to(device)
    student = trainer.fit_student(
        student, teacher, train_loader, val_loader, epochs=Config.STUDENT_EPOCHS
    )

    return student, depth_stats


def run_inference(student_model, depth_stats, device):
    """
    Demonstrates inference on test data and submission file generation.
    """
    print("\n--- Running Inference Demo ---")

    # Load Test Data
    test_loader, test_ids = get_test_loader(depth_stats, load_cached_data=False)

    student_model.eval()
    submission_rows = []

    print(f"Predicting on {len(test_ids)} test images...")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward pass
            logits, _ = student_model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Post-processing
            for i in range(len(probs)):
                # 1. Unpad to original 101x101
                prob_map = unpad_image(
                    probs[i, 0], original_shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)
                )

                # 2. Thresholding (using 0.5 for demo)
                mask = (prob_map > 0.5).astype(np.uint8)

                # 3. RLE Encoding
                rle = rle_encode(mask)
                submission_rows.append(rle)

    # Verify alignment
    if len(submission_rows) != len(test_ids):
        raise AssertionError("Mismatch between number of predictions and test IDs.")

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "rle_mask": submission_rows})

    output_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print("Head of submission:")
    print(submission_df.head())


if __name__ == "__main__":
    # 1. Set Seed
    seed_everything(42)

    # 2. Determine Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Setup Config and Data
    setup_demo_config()

    # 4. Verify Components
    verify_utilities()
    verify_models(device)

    # 5. Run Training
    trained_student, d_stats = run_training_pipeline(device)

    # 6. Run Inference
    run_inference(trained_student, d_stats, device)

    print("\n=== Demonstration Completed Successfully ===")
