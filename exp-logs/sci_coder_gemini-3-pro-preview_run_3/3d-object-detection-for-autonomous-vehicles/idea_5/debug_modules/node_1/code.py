import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    DEVICE,
    NUM_CLASSES,
    GRID_SIZE,
    DOWN_RATIO,
    BATCH_SIZE,
    set_deterministic,
)
from library.dataset import LidarDataset, collate_fn
from library.model import PillarFeatureNet, Backbone, CenterHead, CenterPointNet
from library.utils import decode_predictions
from library.engine import train_one_epoch, LossWrapper


def run_demonstration():
    print("Setting deterministic behavior...")
    set_deterministic(seed=42)

    # ==============================================================================
    # 1. Dataset Instantiation & Verification
    # ==============================================================================
    print("\n--- 1. Dataset Verification ---")
    # Load a tiny subset of the training data for demonstration speed
    subset_size = 8
    dataset = LidarDataset(
        metadata_path=TRAIN_METADATA_PATH, mode="train", num_samples=subset_size
    )

    print(f"Dataset loaded with {len(dataset)} samples.")

    # Fetch one sample
    sample = dataset[0]
    required_keys = [
        "points",
        "gt_boxes",
        "heatmap",
        "dim_map",
        "rot_map",
        "reg_map",
        "z_map",
    ]

    # Verify keys exist
    for key in required_keys:
        if key not in sample:
            raise AssertionError(f"Missing key in dataset sample: {key}")

    # Verify shapes
    # Points: (N, 4) -> x, y, z, intensity
    assert (
        sample["points"].ndim == 2 and sample["points"].shape[1] == 4
    ), f"Points shape mismatch. Expected (N, 4), got {sample['points'].shape}"

    # Heatmap: (NumClasses, H_out, W_out)
    expected_h = GRID_SIZE[1] // DOWN_RATIO
    expected_w = GRID_SIZE[0] // DOWN_RATIO
    assert sample["heatmap"].shape == (
        NUM_CLASSES,
        expected_h,
        expected_w,
    ), f"Heatmap shape mismatch. Expected {(NUM_CLASSES, expected_h, expected_w)}, got {sample['heatmap'].shape}"

    print("Dataset sample verification passed.")

    # ==============================================================================
    # 2. DataLoader & Collate Function
    # ==============================================================================
    print("\n--- 2. DataLoader & Collate Verification ---")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    # Verify batch structure
    # 'points' should be a list of tensors (since point clouds vary in size)
    assert isinstance(batch["points"], list), "Batch 'points' should be a list."
    assert len(batch["points"]) == 2, "Batch size mismatch."

    # 'heatmap' should be a stacked tensor: (B, C, H, W)
    assert batch["heatmap"].shape == (
        2,
        NUM_CLASSES,
        expected_h,
        expected_w,
    ), "Batch heatmap shape incorrect."

    print("DataLoader batch verification passed.")

    # ==============================================================================
    # 3. Component-wise Model Verification
    # ==============================================================================
    print("\n--- 3. Component-wise Model Verification ---")

    # Move batch data to device
    points_batch = [p.to(DEVICE) for p in batch["points"]]

    # A. PillarFeatureNet (VFE)
    vfe = PillarFeatureNet().to(DEVICE)
    vfe_out = vfe(points_batch)
    print(f"VFE Output Shape: {vfe_out.shape}")

    # Expected: (B, 64, 1024, 1024) based on default config
    assert vfe_out.shape[0] == 2
    assert vfe_out.shape[1] == 64
    assert vfe_out.shape[2] == GRID_SIZE[1]
    assert vfe_out.shape[3] == GRID_SIZE[0]

    # B. Backbone
    backbone = Backbone(in_channels=64, out_channels=256).to(DEVICE)
    backbone_out = backbone(vfe_out)
    print(f"Backbone Output Shape: {backbone_out.shape}")

    # Expected: (B, 256, 256, 256) -> Downsampled by 4
    assert backbone_out.shape[1] == 256
    assert backbone_out.shape[2] == expected_h
    assert backbone_out.shape[3] == expected_w

    # C. CenterHead
    head = CenterHead(in_channels=256, num_classes=NUM_CLASSES).to(DEVICE)
    hm, dim, rot, reg, z_map = head(backbone_out)
    print(f"Head Heatmap Shape: {hm.shape}")

    assert hm.shape == (2, NUM_CLASSES, expected_h, expected_w)
    assert dim.shape == (2, 3, expected_h, expected_w)

    print("Component verification passed.")

    # ==============================================================================
    # 4. Full Model & Loss Verification
    # ==============================================================================
    print("\n--- 4. Full Model & Loss Verification ---")

    model = CenterPointNet().to(DEVICE)
    criterion = LossWrapper().to(DEVICE)

    # Forward Pass
    preds = model({"points": points_batch})

    # Prepare targets for loss
    targets = {
        "heatmap": batch["heatmap"].to(DEVICE),
        "dim": batch["dim"].to(DEVICE),
        "rot": batch["rot"].to(DEVICE),
        "reg": batch["reg"].to(DEVICE),
        "z_map": batch["z_map"].to(DEVICE),
        "ind": batch["ind"].to(DEVICE),
        "mask": batch["mask"].to(DEVICE),
    }

    # Calculate Loss
    loss, stats = criterion(preds, targets)

    print(f"Total Loss: {loss.item():.4f}")
    print(f"Loss Stats: {stats}")

    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."

    print("Full model forward and loss calculation passed.")

    # ==============================================================================
    # 5. Decoding Predictions
    # ==============================================================================
    print("\n--- 5. Decoding Predictions ---")

    # Decode the raw output from the model
    # K=5 top predictions
    detections = decode_predictions(
        preds["heatmap"], preds["dim"], preds["rot"], preds["reg"], preds["z_map"], K=5
    )

    print(f"Detections Shape: {detections.shape}")
    # Expected: (B, K, 9) -> [x, y, z, w, l, h, yaw, score, class_id]
    assert detections.shape == (2, 5, 9)

    # Check if coordinates are within reasonable bounds (roughly within point cloud range)
    # Just checking x (index 0)
    x_coords = detections[..., 0]
    assert (x_coords >= -100).all() and (
        x_coords <= 100
    ).all(), "Detected X coordinates out of reasonable bounds."

    print("Prediction decoding passed.")

    # ==============================================================================
    # 6. Minimal Training Loop Demonstration
    # ==============================================================================
    print("\n--- 6. Minimal Training Loop ---")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run 1 epoch
    print("Running 1 epoch of training on subset...")
    train_stats = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,  # Skip scheduler for this short demo
        criterion=criterion,
        device=DEVICE,
        epoch_idx=0,
    )

    print("Training loop execution successful.")
    print(f"Final Train Stats: {train_stats}")


if __name__ == "__main__":
    run_demonstration()
