import torch
import torch.optim as optim
import numpy as np
import os
import sys

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    weighted_boxes_fusion,
    get_image_prediction_string,
)
from library.dataset import get_dataloader
from library.model import SwinDyHeadNet
from library.loss import Criterion
from library.engine import Engine


def main():
    print("=== Starting Demonstration of SwinDyHead Pipeline ===")

    # 1. Setup and Configuration Overrides for Demo
    # We modify the Config class attributes directly to run a fast demo
    seed_everything(Config.SEED)

    print("Configuring for fast demonstration...")
    Config.DEBUG = True  # Use small subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Verification
    print("\n--- Verifying Data Loading ---")
    # Initialize dataloaders
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, debug=True, num_workers=0
    )
    val_loader = get_dataloader(
        split="val", batch_size=Config.BATCH_SIZE, debug=True, num_workers=0
    )

    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")

    # Fetch a single batch to verify structure
    images, targets = next(iter(train_loader))

    # Verify Image Tensor
    # Expected: [Batch_Size, 3, IMG_SIZE, IMG_SIZE]
    assert images.dim() == 4, "Images should be a 4D tensor"
    assert (
        images.shape[0] == Config.BATCH_SIZE
    ), f"Batch size should be {Config.BATCH_SIZE}"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == Config.IMG_SIZE and images.shape[3] == Config.IMG_SIZE
    ), f"Images should be resized to {Config.IMG_SIZE}x{Config.IMG_SIZE}"
    print(f"Image batch shape verified: {images.shape}")

    # Verify Targets
    # Expected: List of dicts with 'boxes', 'labels', 'study_label'
    assert len(targets) == Config.BATCH_SIZE, "Targets list length matches batch size"
    sample_target = targets[0]
    assert "boxes" in sample_target, "Target must contain 'boxes'"
    assert "labels" in sample_target, "Target must contain 'labels'"
    assert "study_label" in sample_target, "Target must contain 'study_label'"

    if len(sample_target["boxes"]) > 0:
        assert (
            sample_target["boxes"].dim() == 2 and sample_target["boxes"].shape[1] == 4
        ), "Boxes should be [N, 4]"
    print("Target structure verified.")

    # 3. Model Initialization and Forward Pass
    print("\n--- Verifying Model Architecture ---")
    model = SwinDyHeadNet().to(device)

    # Move batch to device
    images = images.to(device)
    targets_device = [
        {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
        for t in targets
    ]

    # Forward Pass
    print("Running forward pass...")
    outputs = model(images)

    # Verify Output Keys
    expected_keys = [
        "cls_logits",
        "bbox_preds",
        "anchors",
        "num_anchors_per_level",
        "study_logits",
    ]
    for key in expected_keys:
        assert key in outputs, f"Model output missing key: {key}"

    # Verify Shapes
    # cls_logits: [B, N_anchors, 1] (since NUM_CLASSES_DET=1)
    cls_logits = outputs["cls_logits"]
    assert cls_logits.shape[0] == Config.BATCH_SIZE
    assert cls_logits.shape[2] == Config.NUM_CLASSES_DET

    # study_logits: [B, 4]
    study_logits = outputs["study_logits"]
    assert study_logits.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES_STUDY)

    print(
        f"Model output shapes verified. Anchors generated: {outputs['anchors'].shape[0]}"
    )

    # 4. Loss Calculation
    print("\n--- Verifying Loss Calculation ---")
    criterion = Criterion().to(device)

    loss_dict = criterion(outputs, targets_device)

    # Verify Loss Dictionary
    assert "loss" in loss_dict
    assert "loss_cls" in loss_dict
    assert "loss_box" in loss_dict
    assert "loss_study" in loss_dict

    total_loss = loss_dict["loss"]
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert total_loss > 0, "Loss should be positive"

    print(f"Loss calculation successful. Total Loss: {total_loss.item():.4f}")

    # 5. Training Engine Verification
    print("\n--- Verifying Training Engine ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    engine = Engine(model, optimizer, device)

    print("Running training for one epoch (subset)...")
    avg_loss = engine.train_one_epoch(train_loader, epoch=0)
    assert avg_loss > 0, "Average training loss should be positive"

    print("Running evaluation (subset)...")
    val_map, val_acc = engine.evaluate(val_loader)
    print(f"Evaluation complete. mAP: {val_map:.4f}, Study Acc: {val_acc:.4f}")

    # 6. Utility Functions Verification
    print("\n--- Verifying Utilities ---")

    # Test Weighted Boxes Fusion (WBF)
    # Simulate 2 models predicting the same box
    box1 = [100, 100, 200, 200]
    box2 = [102, 102, 202, 202]  # Slightly shifted
    boxes_list = [[box1], [box2]]  # [Models, Boxes, 4]
    scores_list = [[0.9], [0.8]]
    labels_list = [[1], [1]]

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=[1, 1], iou_thr=0.5
    )

    assert len(fused_boxes) == 1, "WBF should fuse overlapping boxes"
    assert fused_scores[0] > 0.8, "Fused score should be reasonable"
    print("Weighted Boxes Fusion verified.")

    # Test Prediction String Formatting
    pred_str = get_image_prediction_string(fused_boxes, fused_scores)
    assert isinstance(pred_str, str)
    assert "opacity" in pred_str
    print(f"Prediction string format verified: {pred_str[:30]}...")

    print("\n=== Demonstration Complete: All components verified successfully ===")


if __name__ == "__main__":
    main()
