import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 1. Suppress Progress Bars (Monkeypatch tqdm)
# We do this before importing library modules that use tqdm
import tqdm


def nop(it, *args, **kwargs):
    return it


tqdm.tqdm = nop

# 2. Import Library Modules
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, get_best_threshold
from library.dataset import load_dataset_arrays, SaltDataset, get_transforms
from library.models import build_model
from library.losses import CompositeLoss
from library.engine import (
    train_one_epoch,
    evaluate,
    generate_pseudo_labels,
    predict_and_submit,
)


def main():
    print("Starting Salt Segmentation Pipeline Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for speed
    Config.DEBUG_SAMPLE_SIZE = 16  # Tiny dataset
    Config.BATCH_SIZE = 4
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE3 = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set device and seed
    device = Config.DEVICE
    set_seed(Config.SEED)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Utility Verification (RLE)
    # -------------------------------------------------------------------------
    print("\nVerifying Utilities...")
    # Create a dummy 101x101 mask with a square in the middle
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[40:60, 40:60] = 1

    # Encode
    rle_str = rle_encode(dummy_mask)
    assert isinstance(rle_str, str) and len(rle_str) > 0, "RLE encoding failed"

    # Decode
    decoded_mask = rle_decode(rle_str, shape=(101, 101))
    assert np.array_equal(dummy_mask, decoded_mask), "RLE round-trip failed"
    print("RLE Encode/Decode verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading Data...")
    # Load training data (subset)
    train_imgs, train_masks, train_depths, train_ids = load_dataset_arrays(
        Config.TRAIN_METADATA_PATH,
        cache_prefix="demo_train",
        load_cached_data=False,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    assert len(train_imgs) == Config.DEBUG_SAMPLE_SIZE
    assert train_imgs.shape == (Config.DEBUG_SAMPLE_SIZE, 101, 101)

    # Create Dataset and DataLoader
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transforms=get_transforms(mode="train"),
        mode="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Batch
    batch = next(iter(train_loader))
    img_batch = batch["image"]
    mask_batch = batch["mask"]
    depth_batch = batch["depth"]

    # Expected shape after padding to 128x128
    assert img_batch.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Unexpected image shape: {img_batch.shape}"
    assert mask_batch.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Unexpected mask shape: {mask_batch.shape}"
    assert depth_batch.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Unexpected depth shape: {depth_batch.shape}"
    print("Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\nInitializing Models...")

    # Teacher Model (Requires Depth)
    teacher_model = build_model(stage="teacher").to(device)

    # Student Model (Auxiliary Depth Output)
    student_model = build_model(stage="student").to(device)

    # Move batch to device
    img_dev = img_batch.to(device)
    depth_dev = depth_batch.to(device)

    # Test Teacher Forward
    with torch.no_grad():
        teacher_out = teacher_model(img_dev, depth_dev)
        assert teacher_out.shape == (
            Config.BATCH_SIZE,
            1,
            128,
            128,
        ), "Teacher output shape mismatch"

    # Test Student Forward
    with torch.no_grad():
        student_out = student_model(img_dev)
        assert isinstance(student_out, dict), "Student should return dict"
        assert student_out["mask"].shape == (
            Config.BATCH_SIZE,
            1,
            128,
            128,
        ), "Student mask shape mismatch"
        assert student_out["depth"].shape == (
            Config.BATCH_SIZE,
            1,
        ), "Student depth shape mismatch"

    print("Model Forward Pass verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop (Teacher Stage 1)
    # -------------------------------------------------------------------------
    print("\nRunning Training Loop (Teacher)...")
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=Config.LEARNING_RATE)

    loss = train_one_epoch(
        teacher_model, train_loader, optimizer, device, epoch=1, mode="teacher"
    )

    assert not np.isnan(loss), "Training loss is NaN"
    print(f"Teacher Train Loss: {loss:.4f}")

    # -------------------------------------------------------------------------
    # 6. Evaluation Loop
    # -------------------------------------------------------------------------
    print("\nRunning Evaluation...")
    # Use same loader for simplicity as val
    score, val_loss, preds, targets = evaluate(
        teacher_model, train_loader, device, mode="teacher"
    )

    assert preds.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        101,
        101,
    ), "Prediction shape mismatch (should be cropped back)"
    print(f"Validation mAP: {score:.4f}")

    # -------------------------------------------------------------------------
    # 7. Pseudo-Label Generation (Marginalization)
    # -------------------------------------------------------------------------
    print("\nGenerating Pseudo-Labels...")
    # Load test data subset
    test_imgs, test_masks, test_depths, test_ids = load_dataset_arrays(
        Config.TEST_METADATA_PATH,
        cache_prefix="demo_test",
        load_cached_data=False,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    test_dataset = SaltDataset(
        test_imgs,
        test_masks,
        test_depths,
        test_ids,
        transforms=get_transforms(mode="test"),  # Minimal transforms
        mode="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Generate
    pseudo_labels = generate_pseudo_labels(
        [teacher_model], test_loader, device, load_cached_data=False  # List of teachers
    )

    # Note: generate_pseudo_labels returns labels for the *entire* metadata file if it reads from CSV,
    # but here we limited the loader. The function implementation in `engine.py` iterates the loader
    # and fills a dict, then maps to the full test CSV.
    # Since we only processed a subset, the rest will be zeros.
    # We check if at least the processed ones are valid.

    # Check shape (Should match full test set size defined in metadata, which is 1000)
    # But wait, we are running a demo. The `generate_pseudo_labels` function reads Config.TEST_METADATA_PATH.
    # To avoid mismatch, we should assert that the output array length matches the test metadata length.
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    assert len(pseudo_labels) == len(
        full_test_df
    ), "Pseudo labels length mismatch with Test Metadata"
    assert pseudo_labels.shape[1:] == (101, 101), "Pseudo label spatial dim mismatch"
    print("Pseudo-Label generation verification passed.")

    # -------------------------------------------------------------------------
    # 8. Student Training & Submission
    # -------------------------------------------------------------------------
    print("\nSimulating Student Inference & Submission...")

    # We will skip actual Student training (distillation) to save time and just run inference/submission
    # using the initialized student model.

    # Create a dummy val loader for threshold optimization
    val_loader = train_loader

    predict_and_submit(student_model, test_loader, val_loader, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    # The submission file should contain rows for the test set.
    # Since we used a subset loader for inference, `predict_and_submit` logic (which iterates loader)
    # will only populate results for those IDs. The rest default to empty masks.
    # We just verify the file structure.
    assert "id" in sub_df.columns and "rle_mask" in sub_df.columns
    print("Submission verification passed.")

    print("\nAll pipeline components verified successfully.")


if __name__ == "__main__":
    main()
