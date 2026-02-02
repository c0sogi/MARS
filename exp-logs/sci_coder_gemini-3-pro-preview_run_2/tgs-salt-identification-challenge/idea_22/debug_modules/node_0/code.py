import os
import sys
import numpy as np
import torch
import pandas as pd
import cv2
from torch.utils.data import Subset, DataLoader
import torch.optim as optim

# Import provided library modules
from library import config, utils, models, dataset, losses, engine


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Setup Reproducibility
    config.setup_reproducibility(seed=42)
    device = config.DEVICE
    print(f"Device: {device}")

    # 2. Override Config for Speed
    # We reduce batch size and epochs to run a quick demo
    config.BATCH_SIZE = 4
    config.NUM_EPOCHS_TEACHER = 1
    config.NUM_EPOCHS_STUDENT = 1

    # =========================================================================
    # Part 1: Verify Utilities
    # =========================================================================
    print("\n[1/6] Verifying Utilities...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a square of 1s

    rle_str = utils.rle_encode(dummy_mask)
    decoded_mask = utils.rle_decode(rle_str, shape=(101, 101))

    assert np.array_equal(dummy_mask, decoded_mask), "RLE Decode mismatch!"
    print("  -> RLE Encoding/Decoding passed.")

    # Test Padding/Unpadding
    dummy_img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
    padded_img = utils.pad_image(dummy_img, target_size=128)
    assert padded_img.shape == (128, 128), f"Padding shape mismatch: {padded_img.shape}"

    unpadded_img = utils.unpad_image(padded_img, original_size=101)
    assert unpadded_img.shape == (
        101,
        101,
    ), f"Unpadding shape mismatch: {unpadded_img.shape}"
    assert np.array_equal(dummy_img, unpadded_img), "Unpadded image content mismatch!"
    print("  -> Image Padding/Unpadding passed.")

    # Test IoU Calculation
    # Pred: Half overlap
    pred_batch = torch.zeros((2, 10, 10))
    pred_batch[0, 0:5, 0:5] = 1.0  # Image 0: Top-left square

    target_batch = torch.zeros((2, 10, 10))
    target_batch[0, 0:5, 0:5] = 1.0  # Image 0: Perfect match

    ious = utils.calc_iou_batch(pred_batch, target_batch)
    # Image 0: IoU should be 1.0
    # Image 1: Both empty, IoU should be 1.0 (as per metric definition in utils)
    assert np.isclose(ious[0], 1.0), f"IoU mismatch for match: {ious[0]}"
    assert np.isclose(ious[1], 1.0), f"IoU mismatch for empty: {ious[1]}"
    print("  -> IoU Calculation passed.")

    # =========================================================================
    # Part 2: Data Loading & Subsetting
    # =========================================================================
    print("\n[2/6] Loading Data & Creating Subsets...")

    # Load full dataloaders (this triggers caching if not present)
    # We use num_workers=0 to avoid multiprocessing overhead in this quick script
    train_loader_full, val_loader_full, test_loader_full = dataset.get_dataloaders(
        load_cached_data=True, batch_size=config.BATCH_SIZE, num_workers=0
    )

    # Create Subsets (e.g., 16 samples = 4 batches) for speed
    subset_indices = list(range(16))

    train_subset = Subset(train_loader_full.dataset, subset_indices)
    val_subset = Subset(val_loader_full.dataset, subset_indices)
    test_subset = Subset(test_loader_full.dataset, subset_indices)

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"  -> Train Subset Size: {len(train_subset)}")
    print(f"  -> Val Subset Size: {len(val_subset)}")
    print(f"  -> Test Subset Size: {len(test_subset)}")

    # =========================================================================
    # Part 3: Model Initialization & Forward Pass
    # =========================================================================
    print("\n[3/6] Initializing Models...")

    teacher_model = models.PrivilegedTeacher().to(device)
    student_model = models.MultiTaskStudent().to(device)

    # Fetch one batch
    images, masks, depths, ids = next(iter(train_loader))
    images, masks, depths = images.to(device), masks.to(device), depths.to(device)

    print(f"  -> Input Batch Shape: {images.shape}")

    # Teacher Forward
    teacher_logits = teacher_model(images, depths)
    assert teacher_logits.shape == (
        config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Teacher output shape mismatch: {teacher_logits.shape}"

    # Student Forward
    student_logits, student_depth = student_model(images)
    assert student_logits.shape == (
        config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Student mask output shape mismatch: {student_logits.shape}"
    assert student_depth.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Student depth output shape mismatch: {student_depth.shape}"

    print("  -> Forward pass checks passed.")

    # =========================================================================
    # Part 4: Loss Calculation
    # =========================================================================
    print("\n[4/6] Verifying Loss Functions...")

    # Segmentation Loss (Teacher)
    seg_loss_fn = losses.SegmentationLoss()
    loss_t = seg_loss_fn(teacher_logits, masks)
    assert not torch.isnan(loss_t), "Teacher loss is NaN"
    assert loss_t.item() >= 0, "Teacher loss is negative"

    # Student Loss
    student_loss_fn = losses.StudentLoss()
    loss_s, components = student_loss_fn(
        student_logits, teacher_logits, student_depth, masks, depths
    )
    assert not torch.isnan(loss_s), "Student loss is NaN"
    print(f"  -> Teacher Loss: {loss_t.item():.4f}")
    print(
        f"  -> Student Loss: {loss_s.item():.4f} "
        f"(Seg: {components['loss_seg']:.2f}, Distill: {components['loss_distill']:.2f})"
    )

    # =========================================================================
    # Part 5: Training Loops
    # =========================================================================
    print("\n[5/6] Running Training Loops (Demo)...")

    # Teacher Training
    optimizer_t = optim.Adam(teacher_model.parameters(), lr=config.LEARNING_RATE)
    teacher_save_path = os.path.join(config.WORKING_DIR, "demo_teacher.pth")

    print("  -> Fitting Teacher...")
    engine.fit_teacher(
        teacher_model,
        train_loader,
        val_loader,
        optimizer_t,
        seg_loss_fn,
        device,
        epochs=config.NUM_EPOCHS_TEACHER,
        save_path=teacher_save_path,
    )

    # Student Training
    # We load the "best" teacher (current state for demo) to distill from
    teacher_model.eval()
    optimizer_s = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
    student_save_path = os.path.join(config.WORKING_DIR, "demo_student.pth")

    print("  -> Fitting Student...")
    engine.fit_student(
        student_model,
        teacher_model,
        train_loader,
        val_loader,
        optimizer_s,
        student_loss_fn,
        device,
        epochs=config.NUM_EPOCHS_STUDENT,
        save_path=student_save_path,
    )

    assert os.path.exists(teacher_save_path), "Teacher model not saved."
    assert os.path.exists(student_save_path), "Student model not saved."

    # =========================================================================
    # Part 6: Inference & Submission
    # =========================================================================
    print("\n[6/6] Inference & Submission...")

    # Optimize Threshold
    best_threshold = engine.optimize_threshold(student_model, val_loader, device)

    # Generate Predictions on Test Subset
    student_model.eval()
    submission_data = []

    pad_top = (config.IMG_SIZE - config.ORIG_SIZE) // 2
    pad_left = (config.IMG_SIZE - config.ORIG_SIZE) // 2
    h_end = pad_top + config.ORIG_SIZE
    w_end = pad_left + config.ORIG_SIZE

    print("  -> Generating predictions...")
    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)
            logits, _ = student_model(images)
            probs = torch.sigmoid(logits)

            # Unpad
            probs_cropped = probs[:, :, pad_top:h_end, pad_left:w_end]

            # Binarize
            preds = (probs_cropped > best_threshold).cpu().numpy().astype(np.uint8)

            # Encode
            for i in range(len(ids)):
                mask = preds[i, 0]  # (H, W)
                rle = utils.rle_encode(mask)
                submission_data.append({"id": ids[i], "rle_mask": rle})

    # Create DataFrame
    sub_df = pd.DataFrame(submission_data)
    demo_sub_path = os.path.join(config.WORKING_DIR, "submission_demo.csv")
    sub_df.to_csv(demo_sub_path, index=False)

    print(f"  -> Submission saved to {demo_sub_path}")
    print(f"  -> Rows: {len(sub_df)}")
    print(f"  -> Head:\n{sub_df.head()}")

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
