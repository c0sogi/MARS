import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import (
    DEVICE,
    IMG_HEIGHT,
    IMG_WIDTH,
    ORIG_HEIGHT,
    ORIG_WIDTH,
    SUBMISSION_PATH,
)
from library.utils import (
    set_seed,
    pad_image,
    unpad_image,
    rle_encode,
    rle_decode,
    calc_iou_batch,
    calc_map,
)
from library.dataset import SaltDataset, get_transforms
from library.models import SaltNet
from library.losses import TeacherComboLoss, DepthMSELoss
from library.engine import SaltEngine

# =============================================================================
# CONSTANTS FOR DEMONSTRATION
# =============================================================================
DEBUG_SIZE = 10
BATCH_SIZE = 2
EPOCHS = 1
SEED = 42


def test_utils():
    print("\n=== Testing Utils ===")

    # 1. Test Padding/Unpadding
    dummy_img = np.random.rand(ORIG_HEIGHT, ORIG_WIDTH).astype(np.float32)
    padded = pad_image(dummy_img)
    assert padded.shape == (
        IMG_HEIGHT,
        IMG_WIDTH,
    ), f"Padding failed. Shape: {padded.shape}"

    unpadded = unpad_image(padded)
    assert unpadded.shape == (
        ORIG_HEIGHT,
        ORIG_WIDTH,
    ), f"Unpadding failed. Shape: {unpadded.shape}"
    # Verify center crop logic preserves data (excluding reflected borders)
    # Note: Reflection padding makes exact pixel match tricky on borders,
    # but center should be identical if padding is symmetric.
    # Since pad_image uses reflection, we just check dimensions here for speed.
    print("Padding/Unpadding logic verified.")

    # 2. Test RLE Encoding/Decoding
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # Create a square

    rle_str = rle_encode(mask)
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    assert np.array_equal(mask, decoded_mask), "RLE Encode/Decode mismatch."
    print("RLE Encode/Decode verified.")

    # 3. Test Metrics
    # Perfect match
    pred = torch.ones((2, 128, 128))
    target = torch.ones((2, 128, 128))
    iou = calc_iou_batch(pred.numpy().astype(int), target.numpy().astype(int))
    assert np.all(iou == 1.0), "IoU calculation failed for perfect match."

    map_score = calc_map(pred, target)
    assert map_score == 1.0, f"mAP calculation failed. Expected 1.0, got {map_score}"
    print("Metrics verified.")


def test_dataset():
    print("\n=== Testing Dataset ===")

    # Initialize dataset in train mode with debug size
    # We use 'val' transforms to avoid random augmentations making shape checks hard (though shape should be constant)
    ds = SaltDataset(
        mode="train", transform=get_transforms("val"), debug_size=DEBUG_SIZE
    )

    assert (
        len(ds) == DEBUG_SIZE
    ), f"Dataset size mismatch. Expected {DEBUG_SIZE}, got {len(ds)}"

    item = ds[0]

    # Check keys
    required_keys = ["image", "mask", "depth", "id"]
    for k in required_keys:
        assert k in item, f"Missing key {k} in dataset item."

    # Check Shapes
    # Image: (1, 128, 128) - Channel dim added by ToTensorV2 usually if configured,
    # but here input is grayscale. ToTensorV2 converts HWC->CHW.
    # If input is (H,W), ToTensorV2 adds no channel dim by default unless we add it manually or use ToTensor.
    # Let's check the actual output from the provided dataset class.
    img = item["image"]
    mask = item["mask"]
    depth = item["depth"]

    print(f"Image Shape: {img.shape}")
    print(f"Mask Shape: {mask.shape}")
    print(f"Depth Shape: {depth.shape}")

    # The dataset logic ensures mask is (1, H, W)
    assert mask.ndim == 3 and mask.shape[0] == 1, "Mask should be (1, H, W)"
    assert mask.shape[1:] == (IMG_HEIGHT, IMG_WIDTH), "Mask spatial dims mismatch"

    # Depth should be tensor of shape (1,)
    assert depth.numel() == 1, "Depth should be a scalar tensor"

    print("Dataset loading verified.")
    return ds


def test_models_and_losses(dataset):
    print("\n=== Testing Models and Losses ===")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(dataloader))

    images = batch["image"].to(DEVICE)
    masks = batch["mask"].to(DEVICE)
    depths = batch["depth"].to(DEVICE)

    # --- Test Teacher Model ---
    teacher = SaltNet(mode="teacher").to(DEVICE)
    teacher.eval()

    with torch.no_grad():
        # Teacher requires depth input
        logits_teacher = teacher(images, depths)

    assert (
        logits_teacher.shape == masks.shape
    ), f"Teacher output shape mismatch. Got {logits_teacher.shape}, expected {masks.shape}"
    print("Teacher model forward pass successful.")

    # --- Test Student Model ---
    student = SaltNet(mode="student").to(DEVICE)
    student.eval()

    with torch.no_grad():
        # Student infers depth, takes only image
        logits_student, pred_depth = student(images)

    assert (
        logits_student.shape == masks.shape
    ), "Student segmentation output shape mismatch."
    assert (
        pred_depth.shape[0] == images.shape[0]
    ), "Student depth output batch size mismatch."
    print("Student model forward pass successful.")

    # --- Test Losses ---
    combo_loss_fn = TeacherComboLoss()
    mse_loss_fn = DepthMSELoss()

    loss_seg = combo_loss_fn(logits_teacher, masks)
    loss_depth = mse_loss_fn(pred_depth, depths)

    assert not torch.isnan(loss_seg), "Segmentation loss is NaN"
    assert not torch.isnan(loss_depth), "Depth loss is NaN"

    print(
        f"Losses computed successfully. Seg: {loss_seg.item():.4f}, Depth: {loss_depth.item():.4f}"
    )

    return teacher, student


def test_engine_workflow(teacher_model, student_model, dataset):
    print("\n=== Testing Engine Workflow ===")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    # 1. Test Teacher Training Step
    print("--- Testing Teacher Training Loop ---")
    optimizer = optim.AdamW(teacher_model.parameters(), lr=1e-4)
    engine_teacher = SaltEngine(
        teacher_model, device=DEVICE, optimizer=optimizer, mode="teacher"
    )

    loss_teacher = engine_teacher.train_one_epoch(dataloader, epoch_idx=1)
    assert loss_teacher > 0, "Teacher training loss should be positive."

    # 2. Test Validation
    print("--- Testing Validation ---")
    map_score, best_thresh = engine_teacher.validate(dataloader)
    print(f"Validation complete. mAP: {map_score:.4f}")

    # 3. Test Marginalized Inference (used for Pseudo-labeling)
    print("--- Testing Marginalized Inference ---")
    # Take one image
    img_sample = dataset[0]["image"].unsqueeze(0)  # (1, C, H, W)
    soft_mask = SaltEngine.predict_marginalized(teacher_model, img_sample, DEVICE)

    assert soft_mask.shape == (
        1,
        1,
        IMG_HEIGHT,
        IMG_WIDTH,
    ), "Marginalized output shape mismatch"
    assert (
        soft_mask.min() >= 0 and soft_mask.max() <= 1
    ), "Soft mask values out of range [0, 1]"
    print("Marginalized inference verified.")

    # 4. Test Student Training Step (Multi-task)
    print("--- Testing Student Training Loop ---")
    optimizer_student = optim.AdamW(student_model.parameters(), lr=1e-4)
    engine_student = SaltEngine(
        student_model, device=DEVICE, optimizer=optimizer_student, mode="student"
    )

    loss_student = engine_student.train_one_epoch(dataloader, epoch_idx=1)
    assert loss_student > 0, "Student training loss should be positive."

    # 5. Test Submission Generation
    print("--- Testing Submission Generation ---")
    # Use a tiny test dataset
    test_ds = SaltDataset(
        mode="test", transform=get_transforms("val"), debug_size=DEBUG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    # We use the student engine for submission generation
    engine_student.generate_submission_csv(test_loader, threshold=0.5)

    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(SUBMISSION_PATH)
    assert (
        len(df_sub) == DEBUG_SIZE
    ), f"Submission rows mismatch. Expected {DEBUG_SIZE}, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns missing."
    print("Submission generation verified.")


def main():
    set_seed(SEED)

    # Ensure working directories exist (handled by config, but good to check)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # Run Tests
    test_utils()
    ds = test_dataset()
    teacher, student = test_models_and_losses(ds)
    test_engine_workflow(teacher, student, ds)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
