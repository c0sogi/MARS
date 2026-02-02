import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import (
    rle_encode,
    rle_decode,
    pad_image,
    crop_image,
    do_kaggle_metric,
    calculate_iou,
)
from library.dataset import load_data, SaltDataset
from library.model import SpecialistTeacher, GeneralistStudent
from library.losses import LovaszHingeLoss, StudentMultiTaskLoss
from library.engine import (
    set_seed,
    train_one_epoch,
    validate,
    predict_marginalized,
    predict_student,
)


def run_demonstration():
    print("Starting Salt Segmentation Library Demonstration...")

    # =========================================================================
    # 1. Configuration Override for Speed
    # =========================================================================
    print("\n[1] Configuring Environment...")
    # Override Config to run on a tiny subset for demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small batch for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE3 = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist (Config.setup() does this, but good to confirm)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # Set seed
    set_seed(Config.SEED)
    print("    Configuration complete.")

    # =========================================================================
    # 2. Verify Utilities
    # =========================================================================
    print("\n[2] Verifying Utilities...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a square

    rle_str = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "RLE Decode failed to reconstruct original mask"
    print("    RLE Encode/Decode: PASSED")

    # Test Padding/Cropping
    padded_mask = pad_image(dummy_mask, target_size=128)
    assert padded_mask.shape == (
        128,
        128,
    ), f"Padding shape mismatch: {padded_mask.shape}"

    cropped_mask = crop_image(padded_mask, original_size=101)
    assert cropped_mask.shape == (
        101,
        101,
    ), f"Crop shape mismatch: {cropped_mask.shape}"
    assert np.array_equal(
        dummy_mask, cropped_mask
    ), "Crop failed to reconstruct original image"
    print("    Pad/Crop: PASSED")

    # =========================================================================
    # 3. Data Loading
    # =========================================================================
    print("\n[3] Loading Data...")

    # Load Train Data
    # Note: load_data handles caching and depth stats calculation
    train_dataset = load_data("train", load_cached_data=False)
    val_dataset = load_data("val", load_cached_data=False)

    print(f"    Train Dataset Size: {len(train_dataset)}")
    print(f"    Val Dataset Size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Batch Shapes
    images, masks, depths, ids = next(iter(train_loader))

    # Images: (B, 1, 128, 128) - Grayscale, Padded
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Image batch shape error: {images.shape}"
    # Masks: (B, 1, 128, 128)
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Mask batch shape error: {masks.shape}"
    # Depths: (B, 1)
    assert depths.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Depth batch shape error: {depths.shape}"

    print("    Data Shapes: Verified")

    # =========================================================================
    # 4. Teacher Model Training (Stage 1)
    # =========================================================================
    print("\n[4] Stage 1: Specialist Teacher Training...")

    teacher_model = SpecialistTeacher().to(device)
    teacher_loss_fn = LovaszHingeLoss()
    teacher_optimizer = torch.optim.Adam(
        teacher_model.parameters(), lr=Config.LEARNING_RATE
    )

    # Train one epoch
    print("    Training Teacher for 1 epoch...")
    train_loss = train_one_epoch(
        teacher_model,
        train_loader,
        teacher_optimizer,
        device,
        teacher_loss_fn,
        is_student=False,
    )
    print(f"    Teacher Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_map = validate(
        teacher_model, val_loader, device, teacher_loss_fn, is_student=False
    )
    print(f"    Teacher Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}")

    assert not np.isnan(train_loss), "Teacher training loss is NaN"

    # =========================================================================
    # 5. Marginalization / Pseudo-Labeling (Stage 2)
    # =========================================================================
    print("\n[5] Stage 2: Marginalized Inference (Pseudo-labeling)...")

    # We use the teacher to predict on validation set (simulating test set usage)
    # scanning across depths to marginalize uncertainty
    marginalized_preds = predict_marginalized(teacher_model, val_loader, device)

    # Verify output
    sample_id = val_dataset.ids[0]
    sample_pred = marginalized_preds[sample_id]

    assert sample_pred.shape == (
        1,
        128,
        128,
    ), f"Marginalized pred shape error: {sample_pred.shape}"
    assert (
        0.0 <= sample_pred.min() and sample_pred.max() <= 1.0
    ), "Predictions not in [0, 1] range"

    print(f"    Generated pseudo-labels for {len(marginalized_preds)} images.")

    # =========================================================================
    # 6. Student Model Training (Stage 3)
    # =========================================================================
    print("\n[6] Stage 3: Generalist Student Training...")

    student_model = GeneralistStudent().to(device)
    student_loss_fn = StudentMultiTaskLoss(depth_weight=1.0)
    student_optimizer = torch.optim.Adam(
        student_model.parameters(), lr=Config.LEARNING_RATE
    )

    # Train one epoch (Student mode)
    print("    Training Student for 1 epoch...")
    student_train_loss = train_one_epoch(
        student_model,
        train_loader,
        student_optimizer,
        device,
        student_loss_fn,
        is_student=True,
    )
    print(f"    Student Train Loss: {student_train_loss:.4f}")

    # Validate Student
    student_val_loss, student_val_map = validate(
        student_model, val_loader, device, student_loss_fn, is_student=True
    )
    print(
        f"    Student Val Loss: {student_val_loss:.4f}, Val mAP: {student_val_map:.4f}"
    )

    # =========================================================================
    # 7. Final Inference
    # =========================================================================
    print("\n[7] Inference on Test Data...")

    test_dataset = load_data("test", load_cached_data=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"    Test Dataset Size: {len(test_dataset)}")

    # Predict using Student (with TTA enabled in Config)
    test_preds = predict_student(student_model, test_loader, device)

    assert len(test_preds) == len(test_dataset), "Mismatch in prediction count"

    # Demonstrate post-processing for one sample
    sample_test_id = list(test_preds.keys())[0]
    prob_map = test_preds[sample_test_id]

    # Threshold
    binary_mask = (prob_map > 0.5).astype(np.uint8)
    if binary_mask.ndim == 3:
        binary_mask = binary_mask[0]

    # Crop
    final_mask = crop_image(binary_mask, original_size=Config.ORIG_SIZE)

    # Encode
    rle_result = rle_encode(final_mask)

    print(f"    Sample ID: {sample_test_id}")
    print(f"    RLE Result (first 20 chars): {rle_result[:20]}...")
    print("    Inference pipeline verified.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
