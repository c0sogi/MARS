import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, mask_to_boxes, compute_iou, calculate_map
from library.dataset import ChestXrayDataset
from library.model import EfficientNetUNet
from library.engine import train_one_epoch, validate


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print("\n--- 1. Setup and Configuration ---")
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 images for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Utility Functions
    print("\n--- 2. Verifying Utility Functions ---")

    # Test mask_to_boxes
    # Create a 100x100 mask with a 10x10 square at (10, 10)
    dummy_mask = np.zeros((100, 100), dtype=np.float32)
    dummy_mask[10:20, 10:20] = 1.0
    boxes = mask_to_boxes(dummy_mask, threshold=0.5)

    assert len(boxes) == 1, "Should detect exactly one box"
    # Expected: x, y, x2, y2. x=10, y=10, w=10, h=10 -> x2=20, y2=20
    assert boxes[0] == [10, 10, 20, 20], f"Box coordinates mismatch: {boxes[0]}"
    print("mask_to_boxes: Verified")

    # Test compute_iou
    box_a = [10, 10, 20, 20]
    box_b = [10, 10, 20, 20]  # Identical
    box_c = [100, 100, 110, 110]  # No overlap
    box_d = [15, 10, 25, 20]  # 50% overlap horizontally?
    # Area A = 100. Area D = 100. Intersection: x[15, 20], y[10, 20] -> 5 * 10 = 50.
    # Union = 100 + 100 - 50 = 150. IoU = 50/150 = 0.333

    iou_same = compute_iou(box_a, box_b)
    iou_none = compute_iou(box_a, box_c)
    iou_partial = compute_iou(box_a, box_d)

    assert np.isclose(iou_same, 1.0), "IoU for identical boxes should be 1.0"
    assert np.isclose(iou_none, 0.0), "IoU for disjoint boxes should be 0.0"
    assert np.isclose(iou_partial, 1 / 3), f"IoU calculation incorrect: {iou_partial}"
    print("compute_iou: Verified")

    # Test calculate_map (simple case)
    pred_boxes = [{"boxes": [[10, 10, 20, 20]], "scores": [0.9]}]
    true_boxes = [{"boxes": [[10, 10, 20, 20]]}]
    map_score = calculate_map(pred_boxes, true_boxes, iou_threshold=0.5)
    assert np.isclose(map_score, 1.0), "mAP should be 1.0 for perfect prediction"
    print("calculate_map: Verified")

    # 3. Dataset Instantiation and Verification
    print("\n--- 3. Dataset Loading and Verification ---")

    # Initialize Train Dataset
    print("Initializing Train Dataset...")
    train_dataset = ChestXrayDataset(split="train", load_cached_data=False)
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset size mismatch: {len(train_dataset)}"

    # Check one sample
    sample = train_dataset[0]
    img = sample["image"]
    mask = sample["mask"]
    label = sample["label"]

    # Verify Shapes
    # Image: (3, 512, 512) - albumentations ToTensorV2 moves channels first
    assert img.shape == (
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Image shape mismatch: {img.shape}"
    # Mask: (1, 512, 512) - unsqueezed in dataset
    assert mask.shape == (
        1,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Mask shape mismatch: {mask.shape}"
    # Label: (4,)
    assert label.shape == (4,), f"Label shape mismatch: {label.shape}"

    print("Train Dataset: Verified shapes and keys.")

    # Initialize Val Dataset
    print("Initializing Val Dataset...")
    val_dataset = ChestXrayDataset(split="val", load_cached_data=False)
    assert len(val_dataset) > 0
    print("Val Dataset: Verified.")

    # 4. Model Initialization and Forward Pass
    print("\n--- 4. Model Initialization ---")
    model = EfficientNetUNet(num_classes=Config.NUM_STUDY_CLASSES)
    model.to(Config.DEVICE)

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1]).to(
        Config.DEVICE
    )

    # Forward pass
    seg_logits, cls_logits = model(dummy_input)

    # Verify Output Shapes
    # Seg: (B, 1, H, W)
    assert seg_logits.shape == (
        2,
        1,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Seg logits shape mismatch: {seg_logits.shape}"
    # Cls: (B, 4)
    assert cls_logits.shape == (2, 4), f"Cls logits shape mismatch: {cls_logits.shape}"

    print("Model Forward Pass: Verified output shapes.")

    # 5. Training Loop Demonstration
    print("\n--- 5. Training Loop Demonstration ---")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print(
        f"Starting training for {Config.EPOCHS} epoch(s) on {len(train_dataset)} samples..."
    )
    train_metrics = train_one_epoch(
        model, optimizer, train_loader, Config.DEVICE, epoch=1
    )

    assert "loss" in train_metrics
    assert "seg_loss" in train_metrics
    assert "cls_loss" in train_metrics
    print(f"Training completed. Loss: {train_metrics['loss']:.4f}")

    # 6. Validation Demonstration
    print("\n--- 6. Validation Demonstration ---")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    val_metrics = validate(model, val_loader, Config.DEVICE)

    assert "loss" in val_metrics
    assert "map" in val_metrics
    print(
        f"Validation completed. Loss: {val_metrics['loss']:.4f}, mAP: {val_metrics['map']:.4f}"
    )

    # 7. Test Set Inference (Simulation)
    print("\n--- 7. Test Set Inference Simulation ---")
    # We won't run full inference, but verify the dataset loads correctly for test
    test_dataset = ChestXrayDataset(split="test", load_cached_data=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    test_batch = next(iter(test_loader))
    # Test batch keys: image, study_id, image_id, orig_dim
    assert "image" in test_batch
    assert "study_id" in test_batch
    assert "orig_dim" in test_batch

    # Run model in eval mode
    model.eval()
    with torch.no_grad():
        t_images = test_batch["image"].to(Config.DEVICE)
        t_seg, t_cls = model(t_images)

        # Check shapes
        assert t_seg.shape[0] == 2
        assert t_cls.shape[0] == 2

    print("Test Inference: Verified batch loading and model prediction.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
