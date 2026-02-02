import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.dataset import SaltDataset, get_transforms
from library.models import SaltNet
from library.losses import TeacherLoss, StudentLoss
from library.engine import Engine
from library.distillation import generate_marginalized_pseudo_labels, PseudoDataset
from library.utils import rle_encode, rle_decode, pad_image, unpad_image


def main():
    print("Starting Salt Segmentation Pipeline Demo...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    Engine.set_seed(42)
    device = Config.DEVICE

    # Override Config parameters for a fast demonstration run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BATCH_SIZE = 4
    Config.TEACHER_EPOCHS = 1
    Config.STUDENT_EPOCHS = 1
    Config.DEPTH_SCAN_STEPS = 2  # Reduce scan steps for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    Config.setup()

    # Load Metadata
    print("\n[1] Loading Metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Slice datasets to a tiny subset for speed
    train_df_subset = train_df.iloc[:16].copy()
    val_df_subset = val_df.iloc[:8].copy()
    test_df_subset = test_df.iloc[:8].copy()

    # Save sliced test CSV because generate_marginalized_pseudo_labels reads from file
    temp_test_csv = os.path.join(Config.WORKING_DIR, "temp_test_subset.csv")
    test_df_subset.to_csv(temp_test_csv, index=False)
    Config.TEST_CSV = temp_test_csv  # Point Config to the subset file

    print(
        f"    Subset sizes: Train={len(train_df_subset)}, Val={len(val_df_subset)}, Test={len(test_df_subset)}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[2] Initializing Datasets...")

    # Initialize Train Dataset
    train_ds = SaltDataset(
        df=train_df_subset,
        mode="train",
        transform=get_transforms("train"),
        load_cached=False,
        cache_name="demo_train",
    )
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Initialize Validation Dataset
    val_ds = SaltDataset(
        df=val_df_subset,
        mode="val",
        transform=get_transforms("valid"),
        load_cached=False,
        cache_name="demo_val",
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch Shapes
    batch_imgs, batch_masks, batch_depths, batch_ids = next(iter(train_loader))
    print(
        f"    Batch Shapes Verified: Img={batch_imgs.shape}, Mask={batch_masks.shape}, Depth={batch_depths.shape}"
    )

    assert batch_imgs.shape == (Config.BATCH_SIZE, 1, 128, 128), "Incorrect Image Shape"
    assert batch_masks.shape == (Config.BATCH_SIZE, 1, 128, 128), "Incorrect Mask Shape"
    assert batch_depths.shape == (Config.BATCH_SIZE, 1), "Incorrect Depth Shape"

    # -------------------------------------------------------------------------
    # 3. Stage 1: Specialist Teacher Training
    # -------------------------------------------------------------------------
    print("\n[3] Stage 1: Training Specialist Teacher (Depth-Injected)...")

    teacher_model = SaltNet(mode="teacher").to(device)
    teacher_loss_fn = TeacherLoss()
    optimizer_teacher = torch.optim.AdamW(teacher_model.parameters(), lr=1e-3)

    # Train one epoch
    t_loss = Engine.train_teacher_epoch(
        teacher_model, train_loader, optimizer_teacher, device, teacher_loss_fn
    )
    print(f"    Teacher Epoch 1 Loss: {t_loss:.4f}")

    # Validate
    t_val_loss, t_val_map = Engine.validate(
        teacher_model, val_loader, device, teacher_loss_fn, mode="teacher"
    )
    print(f"    Teacher Val Loss: {t_val_loss:.4f}, mAP: {t_val_map:.4f}")

    # Save Checkpoint
    teacher_ckpt_path = os.path.join(Config.WORKING_DIR, "stage1_teacher.pth")
    Engine.save_checkpoint(teacher_model, teacher_ckpt_path)
    assert os.path.exists(teacher_ckpt_path), "Teacher checkpoint not saved"
    print("    Teacher checkpoint saved.")

    # -------------------------------------------------------------------------
    # 4. Stage 2: Marginalized Pseudo-Label Generation
    # -------------------------------------------------------------------------
    print("\n[4] Stage 2: Generating Marginalized Pseudo-Labels for Test Set...")

    # This function loads the teacher model, scans depth range, and averages predictions
    # It uses Config.TEST_CSV which we pointed to our subset
    pseudo_labels = generate_marginalized_pseudo_labels(
        teacher_model_paths=[teacher_ckpt_path], device=device, load_cached_data=False
    )

    # Verify Pseudo-Labels
    assert len(pseudo_labels) == len(test_df_subset), "Mismatch in pseudo-label count"
    sample_id = test_df_subset.iloc[0]["id"]
    assert pseudo_labels[sample_id].shape == (
        128,
        128,
    ), "Incorrect pseudo-label dimensions"
    print(f"    Generated {len(pseudo_labels)} pseudo-labels successfully.")

    # -------------------------------------------------------------------------
    # 5. Stage 3: Generalist Student Training
    # -------------------------------------------------------------------------
    print("\n[5] Stage 3: Training Generalist Student (Multi-Task Distillation)...")

    # Create Unlabeled Dataset using generated pseudo-labels
    unlabeled_ds = PseudoDataset(
        df=test_df_subset,
        pseudo_labels=pseudo_labels,
        transform=get_transforms("train"),
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    student_model = SaltNet(mode="student").to(device)
    student_loss_fn = StudentLoss()
    optimizer_student = torch.optim.AdamW(student_model.parameters(), lr=1e-3)

    # Train one epoch (Labeled + Unlabeled)
    s_loss = Engine.train_student_epoch(
        student_model,
        train_loader,
        unlabeled_loader,
        optimizer_student,
        device,
        student_loss_fn,
    )
    print(f"    Student Epoch 1 Loss: {s_loss:.4f}")

    # Validate Student
    s_val_loss, s_val_map = Engine.validate(
        student_model, val_loader, device, student_loss_fn, mode="student"
    )
    print(f"    Student Val Loss: {s_val_loss:.4f}, mAP: {s_val_map:.4f}")

    # Save Student
    student_ckpt_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Engine.save_checkpoint(student_model, student_ckpt_path)
    print("    Student checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Utilities...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a 10x10 square
    rle_str = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    assert np.array_equal(dummy_mask, decoded_mask), "RLE Decode mismatch"
    print("    RLE Encoding/Decoding verified.")

    # Test Padding Logic
    dummy_img = np.zeros((101, 101), dtype=np.uint8)
    padded = pad_image(dummy_img)
    assert padded.shape == (128, 128), f"Padding failed, got {padded.shape}"

    unpadded = unpad_image(padded)
    assert unpadded.shape == (101, 101), f"Unpadding failed, got {unpadded.shape}"
    print("    Image Padding/Unpadding verified.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
