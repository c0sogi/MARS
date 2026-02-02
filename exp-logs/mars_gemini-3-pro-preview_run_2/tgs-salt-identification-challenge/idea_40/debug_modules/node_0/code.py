import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config
from library.utils import set_seed, pad_image, unpad_image, rle_encode
from library.model import SaltNet
from library.losses import CombinedLoss, AuxiliaryMSELoss, StableBCELoss
from library.data import get_fold_datasets, PseudoDataset, get_test_dataset
from library.train_eval import (
    train_teacher_epoch,
    train_student_epoch,
    validate,
    generate_submission,
)
from library.pseudo_label import generate_marginalized_labels


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("--- 1. Setup and Configuration ---")
    set_seed(42)

    # Override Config for the demo run to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute load
    Config.BATCH_SIZE = 8
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE3 = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small demo to avoid overhead
    Config.AUG_ELASTIC_PROB = 0.0  # Disable heavy augs for speed

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("--- 2. Verifying Utilities ---")

    # Test Padding Logic
    dummy_img = np.zeros((101, 101), dtype=np.uint8)
    padded = pad_image(dummy_img)
    assert padded.shape == (
        128,
        128,
    ), f"Padding failed. Expected (128, 128), got {padded.shape}"

    unpadded = unpad_image(padded)
    assert unpadded.shape == (
        101,
        101,
    ), f"Unpadding failed. Expected (101, 101), got {unpadded.shape}"

    # Test RLE Encoding
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[0, 0] = 1  # Top-left pixel
    mask[1, 0] = 1  # Pixel below top-left
    # RLE is column-major. Index 1 and Index 2 (1-based). Run should be '1 2'
    rle = rle_encode(mask)
    assert isinstance(rle, str), "RLE encode should return a string"
    assert len(rle) > 0, "RLE string should not be empty for non-empty mask"
    print("Utilities verified successfully.")

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("--- 3. Loading Data ---")

    # Load Fold 0 data (this handles caching internally in Config.WORKING_DIR)
    # We set load_cached_data=False to demonstrate processing logic
    train_ds_full, val_ds_full, scaler = get_fold_datasets(
        fold_idx=0, load_cached_data=False
    )

    # Create tiny subsets for demonstration speed
    train_indices = range(16)
    val_indices = range(8)

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify batch structure
    images, masks, depths, ids = next(iter(train_loader))
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Image batch shape mismatch: {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Mask batch shape mismatch: {masks.shape}"
    assert depths.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Depth batch shape mismatch: {depths.shape}"
    print("Data loaded and verified.")

    # ==========================================
    # 4. Teacher Model Training (Stage 1)
    # ==========================================
    print("--- 4. Training Teacher Model (Stage 1) ---")
    device = Config.DEVICE

    # Instantiate Teacher (requires depth injection)
    teacher_model = SaltNet(mode="teacher").to(device)

    # Setup Optimizer and Loss
    optimizer_teacher = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    loss_fn = CombinedLoss()

    # Train for 1 epoch
    print("Running teacher training epoch...")
    train_loss = train_teacher_epoch(
        teacher_model, train_loader, optimizer_teacher, device, loss_fn
    )
    print(f"Teacher Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_map = validate(teacher_model, val_loader, device, loss_fn)
    print(f"Teacher Val Loss: {val_loss:.4f}, mAP: {val_map:.4f}")

    # Save Checkpoint
    teacher_ckpt_path = os.path.join(Config.WORKING_DIR, "teacher_ckpt.pth")
    torch.save(teacher_model.state_dict(), teacher_ckpt_path)
    assert os.path.exists(teacher_ckpt_path), "Teacher checkpoint not saved."

    # ==========================================
    # 5. Pseudo-Label Generation
    # ==========================================
    print("--- 5. Generating Pseudo-Labels ---")

    # Generate soft masks for a small subset of test data (limit_count=10)
    # This uses the saved teacher checkpoint
    soft_masks = generate_marginalized_labels(
        model_paths=[teacher_ckpt_path],
        depth_scaler=scaler,
        load_cached_data=False,
        limit_count=10,
    )

    assert len(soft_masks) > 0, "No pseudo-labels generated."
    sample_id = list(soft_masks.keys())[0]
    assert soft_masks[sample_id].shape == (1, 128, 128), "Soft mask shape mismatch."
    print(f"Generated {len(soft_masks)} pseudo-labels.")

    # ==========================================
    # 6. Student Model Training (Stage 3)
    # ==========================================
    print("--- 6. Training Student Model (Stage 3) ---")

    # Instantiate Student (predicts depth via aux head)
    student_model = SaltNet(mode="student").to(device)

    # Prepare PseudoDataset
    # We need to construct a data dictionary for the specific IDs we have pseudo-labels for
    test_ds_full = get_test_dataset(scaler, load_cached_data=True)

    # Filter test dataset to only include the IDs we generated labels for
    target_ids = set(soft_masks.keys())
    indices = [i for i, x in enumerate(test_ds_full.ids) if x in target_ids]

    pseudo_data_dict = {
        "images": test_ds_full.images[indices],
        "ids": test_ds_full.ids[indices],
        "depths": test_ds_full.depths[indices],
    }

    pseudo_ds = PseudoDataset(
        data_dict=pseudo_data_dict,
        soft_masks_dict=soft_masks,
        transform=None,  # No augmentation for demo
    )

    unlabeled_loader = DataLoader(pseudo_ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Setup Losses and Optimizer
    seg_loss_fn = CombinedLoss()
    aux_loss_fn = AuxiliaryMSELoss()
    soft_loss_fn = StableBCELoss()
    optimizer_student = torch.optim.Adam(student_model.parameters(), lr=1e-3)

    # Train Student (Multi-Task: Labeled + Unlabeled)
    print("Running student training epoch...")
    student_metrics = train_student_epoch(
        student_model,
        train_loader,  # Labeled Data
        unlabeled_loader,  # Unlabeled Data
        optimizer_student,
        device,
        seg_loss_fn,
        aux_loss_fn,
        soft_loss_fn,
    )

    print("Student Metrics:", student_metrics)

    # Validate Student
    val_loss_s, val_map_s = validate(student_model, val_loader, device, seg_loss_fn)
    print(f"Student Val Loss: {val_loss_s:.4f}, mAP: {val_map_s:.4f}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    print("--- 7. Generating Submission ---")

    # Create a loader for the test subset we used
    test_subset = Subset(test_ds_full, indices)
    test_loader_sub = DataLoader(
        test_subset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    generate_submission(student_model, test_loader_sub, device)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())

    assert len(df_sub) == len(indices), "Submission row count mismatch."
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns mismatch."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
