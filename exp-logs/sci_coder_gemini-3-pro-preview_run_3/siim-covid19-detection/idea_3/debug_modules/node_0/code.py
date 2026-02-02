import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

# Import from the provided library
from library import config, utils, dataset, model, loss, train, predict


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Create a temporary directory for this run
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override config paths to point to our demo directory
    config.WORKING_DIR = os.path.join(DEMO_DIR, "cache")
    config.SUBMISSION_DIR = DEMO_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    config.CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Override hyperparameters for speed
    config.EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 2  # Low worker count for small data
    config.IMG_SIZE = 256  # Smaller image size for faster processing in demo

    # Set seeds
    utils.seed_everything(config.SEED)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Subsetting)
    # -------------------------------------------------------------------------
    print("\n[2] Preparing data subsets...")

    # Load original metadata
    orig_train_df = pd.read_csv("./metadata/train.csv")
    orig_val_df = pd.read_csv("./metadata/val.csv")
    orig_test_df = pd.read_csv("./metadata/test.csv")

    # Sample a small subset (e.g., 10 samples for train, 5 for val/test)
    # We ensure we select samples that exist on disk
    demo_train_df = orig_train_df.head(12).copy()
    demo_val_df = orig_val_df.head(4).copy()
    demo_test_df = orig_test_df.head(4).copy()

    # Save these subsets to the demo directory
    demo_train_path = os.path.join(DEMO_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_DIR, "test.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    # Update config to point to these new metadata files
    config.TRAIN_METADATA_PATH = demo_train_path
    config.VAL_METADATA_PATH = demo_val_path
    config.TEST_METADATA_PATH = demo_test_path

    print(
        f"Created subset metadata: Train={len(demo_train_df)}, Val={len(demo_val_df)}, Test={len(demo_test_df)}"
    )

    # -------------------------------------------------------------------------
    # 3. Verify Utils (Box Logic)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Utility Functions...")

    # Test box_to_mask
    h, w = 100, 100
    # Box format: x, y, w, h
    dummy_box_str = "[{'x': 10, 'y': 10, 'width': 20, 'height': 20}]"
    mask = utils.box_to_mask(dummy_box_str, h, w)

    assert mask.shape == (h, w), "Mask shape mismatch"
    assert mask.sum() == 20 * 20, f"Mask area incorrect. Expected 400, got {mask.sum()}"

    # Test mask_to_boxes
    # Note: mask_to_boxes expects [xmin, ymin, xmax, ymax, score]
    boxes = utils.mask_to_boxes(mask, threshold=0.5)
    assert len(boxes) == 1, "Should detect exactly one box"
    b = boxes[0]
    # Allow small tolerance for contour approximation
    assert abs(b[0] - 10) <= 1 and abs(b[1] - 10) <= 1, "Box coordinates mismatch"
    print("Utils verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Dataset & Caching
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Dataset and Caching...")

    # Instantiate dataset (this will trigger caching of the subset)
    ds_train = dataset.SIIMDataset(demo_train_df, split="train", load_cached_data=False)

    # Check length
    assert len(ds_train) == len(demo_train_df)

    # Fetch one item
    sample = ds_train[0]

    # Check keys
    required_keys = ["image", "mask", "label", "study_id", "image_id"]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    # Check shapes
    # Image: (3, IMG_SIZE, IMG_SIZE) - RGB
    assert sample["image"].shape == (
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Image shape mismatch: {sample['image'].shape}"
    # Mask: (1, IMG_SIZE, IMG_SIZE)
    assert sample["mask"].shape == (
        1,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Mask shape mismatch: {sample['mask'].shape}"
    # Label: (4,) - One-hot
    assert sample["label"].shape == (
        4,
    ), f"Label shape mismatch: {sample['label'].shape}"

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Model & Loss
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model and Loss...")

    device = config.DEVICE
    net = model.MultiTaskUNet(pretrained=False).to(
        device
    )  # False for speed/offline safety in demo

    # Create dummy batch
    imgs = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
    masks = (
        torch.randint(0, 2, (2, 1, config.IMG_SIZE, config.IMG_SIZE)).float().to(device)
    )
    labels = torch.randn(2, 4).softmax(dim=1).to(device)  # Dummy one-hot probs

    # Forward pass
    seg_logits, class_logits = net(imgs)

    # Check output shapes
    assert seg_logits.shape == (
        2,
        1,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), "Seg logits shape mismatch"
    assert class_logits.shape == (2, 4), "Class logits shape mismatch"

    # Loss calculation
    criterion = loss.MultiTaskLoss()
    total_loss, metrics = criterion(seg_logits, class_logits, masks, labels)

    assert not torch.isnan(total_loss), "Loss is NaN"
    assert "seg_loss" in metrics and "class_loss" in metrics, "Missing loss metrics"

    print("Model and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 6. Run Training Loop (Integration Test)
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    # We use the provided train.run_training function
    # It will load the metadata from the paths we overrode in config
    try:
        train.run_training(load_cached_data=True, epochs=1, debug=False)
    except Exception as e:
        print(f"Training failed: {e}")
        raise e

    assert os.path.exists(config.CHECKPOINT_PATH), "Checkpoint file was not created."
    print("Training loop completed successfully.")

    # -------------------------------------------------------------------------
    # 7. Run Inference (Integration Test)
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference...")

    # We use the provided predict.generate_submission function
    try:
        predict.generate_submission(load_cached_data=True)
    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert (
        "id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check a few rows
    print("\nSample Submission Rows:")
    print(sub_df.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
