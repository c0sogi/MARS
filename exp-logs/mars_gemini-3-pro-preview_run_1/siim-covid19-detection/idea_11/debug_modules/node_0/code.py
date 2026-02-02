import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from provided library files
from library.utils import (
    seed_everything,
    iou_bbox,
    get_box_from_mask,
    calculate_ap,
    map_calculation,
    parse_prediction_string,
)
from library.dataset import get_dataset, SIIMDataset
from library.model import ResNet18UNet
from library.engine import train_one_epoch, evaluate

# Configuration
SEED = 42
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_section(name):
    print(f"\n{'='*20} {name} {'='*20}")


def test_utils():
    print_section("Testing Utils")

    # 1. Test Seed
    seed_everything(SEED)
    print("Seed set successfully.")

    # 2. Test IoU
    # Box format: [x1, y1, x2, y2]
    box1 = [0, 0, 10, 10]  # Area 100
    box2 = [
        5,
        0,
        15,
        10,
    ]  # Area 100, Intersection 5x10=50. Union = 150. IoU = 50/150 = 1/3
    iou = iou_bbox(box1, box2)
    assert (
        abs(iou - 0.333333) < 1e-5
    ), f"IoU calculation failed. Expected ~0.333, got {iou}"
    print(f"IoU Test Passed: {iou:.4f}")

    # 3. Test get_box_from_mask
    # Create a 100x100 mask with a 10x10 square at (10, 10)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:20, 10:20] = 1.0
    boxes = get_box_from_mask(mask, threshold=0.5)

    assert len(boxes) == 1, "Should find exactly one box"
    b = boxes[0]
    # cv2.boundingRect returns x, y, w, h. get_box_from_mask converts to x1, y1, x2, y2
    # We expect x1=10, y1=10, x2=20, y2=20
    assert b == [
        10,
        10,
        20,
        20,
    ], f"Box extraction failed. Expected [10, 10, 20, 20], got {b}"
    print(f"Box Extraction Test Passed: {boxes}")

    # 4. Test mAP Calculation logic (Synthetic)
    # GT: Image1 has class 'opacity' at [0,0,10,10]
    # Pred: Image1 has class 'opacity' at [0,0,10,10] with conf 0.9
    gt_df = pd.DataFrame(
        {
            "id": ["img1", "img2"],
            "PredictionString": ["opacity 1 0 0 10 10", "none 1 0 0 1 1"],
        }
    )
    pred_df = pd.DataFrame(
        {
            "id": ["img1", "img2"],
            "PredictionString": ["opacity 0.9 0 0 10 10", "none 0.8 0 0 1 1"],
        }
    )

    # Note: map_calculation in utils.py handles multi-class.
    # 'none' is treated as a class if present in the strings.
    # However, standard mAP usually ignores 'none' for detection or treats it specifically.
    # Let's see how the provided util handles it. It parses all labels.

    metric = map_calculation(gt_df, pred_df, iou_threshold=0.5)
    print(f"mAP Calculation Test: {metric:.4f}")
    assert 0.0 <= metric <= 1.0, "mAP should be between 0 and 1"


def test_dataset():
    print_section("Testing Dataset")

    # Use debug=True to load only 100 images
    print("Loading Train Dataset (Debug Mode)...")
    ds_train = get_dataset("train", load_cached_data=False, debug=True)

    assert len(ds_train) > 0, "Dataset should not be empty"
    print(f"Dataset Size: {len(ds_train)}")

    # Fetch one sample
    img, mask, label = ds_train[0]

    # Verify Shapes
    # Image: (3, 512, 512) - Albumentations ToTensorV2 produces (C, H, W)
    # Mask: (1, 512, 512) - Dataset __getitem__ ensures this
    # Label: (4,) - One-hot encoded study label

    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Mask Shape: {mask.shape}")
    print(f"Sample Label Shape: {label.shape}")

    assert img.shape == (3, 512, 512), f"Unexpected image shape: {img.shape}"
    assert mask.shape == (1, 512, 512), f"Unexpected mask shape: {mask.shape}"
    assert label.shape == (4,), f"Unexpected label shape: {label.shape}"

    assert isinstance(img, torch.Tensor), "Image should be a Tensor"
    assert isinstance(mask, torch.Tensor), "Mask should be a Tensor"
    assert isinstance(label, torch.Tensor), "Label should be a Tensor"

    return ds_train


def test_model():
    print_section("Testing Model")

    model = ResNet18UNet(num_classes=4, pretrained=False).to(DEVICE)
    model.eval()

    # Create dummy input: Batch Size 2, 3 Channels, 512x512
    dummy_input = torch.randn(2, 3, 512, 512).to(DEVICE)

    with torch.no_grad():
        mask_out, cls_out = model(dummy_input)

    print(f"Model Mask Output Shape: {mask_out.shape}")
    print(f"Model Class Output Shape: {cls_out.shape}")

    # Expected:
    # Mask: (B, 1, 512, 512) - Segmentation head outputs 1 channel
    # Class: (B, 4) - Classification head outputs 4 classes
    assert mask_out.shape == (2, 1, 512, 512), "Incorrect mask output shape"
    assert cls_out.shape == (2, 4), "Incorrect class output shape"

    return model


def test_engine(model, ds_train):
    print_section("Testing Engine (Training Loop)")

    # Create DataLoader
    loader = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    print("Running train_one_epoch for 1 epoch...")
    # We pass None for scheduler to simplify
    loss = train_one_epoch(model, loader, optimizer, None, scaler, DEVICE)

    print(f"Training Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"
    assert loss > 0, "Training loss should be positive"

    print("\nRunning evaluation...")
    # For evaluation, we need the validation dataframe to match the loader
    # Since we are using the debug train dataset as a proxy for validation to save time loading another DS
    # We need to construct a dataframe that matches ds_train.ids

    # Reconstruct metadata df for the subset
    # get_dataset(debug=True) loads the first 100 rows of the csv
    df_meta = pd.read_csv("./metadata/train.csv").head(100)

    # Ensure the loader is not shuffled for evaluation mapping
    eval_loader = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    val_loss, val_map = evaluate(model, eval_loader, DEVICE, df_meta)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0 <= val_map <= 1, "mAP should be between 0 and 1"


if __name__ == "__main__":
    # Ensure clean working directory for cache if needed,
    # but library uses specific path. We assume it's writable.

    try:
        # 1. Utils
        test_utils()

        # 2. Dataset
        ds = test_dataset()

        # 3. Model
        model = test_model()

        # 4. Engine
        test_engine(model, ds)

        print_section("All Tests Completed Successfully")

    except AssertionError as e:
        print(f"\n!!! ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
