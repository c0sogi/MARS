import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import warnings
import random
import shutil

# Import library modules
from library.config import Config
from library.utils import (
    rle_encode,
    rle_decode,
    calculate_dice,
    calculate_hausdorff,
    keep_largest_component,
    get_gaussian_weight_map,
)
from library.dataset import prepare_data, GIDataset, get_transforms
from library.model import UNetPlusPlus25D
from library.losses import BCETverskyLoss
from library.inference import predict_sliding_window

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_utils():
    print("\n=== Demonstrating Library Utils ===")

    # 1. RLE Encoding/Decoding
    H, W = 100, 100
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # Create a 10x10 square

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, (H, W))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(mask, decoded), "Decoded mask does not match original"
    print("RLE Encode/Decode: Success")

    # 2. Dice Calculation
    dice_perfect = calculate_dice(mask, mask)
    assert np.isclose(
        dice_perfect, 1.0
    ), f"Perfect Dice should be 1.0, got {dice_perfect}"

    mask_empty = np.zeros((H, W), dtype=np.uint8)
    dice_empty = calculate_dice(mask_empty, mask_empty)
    assert dice_empty == 0.0, "Dice of two empty masks should be 0.0 per definition"

    mask_mismatch = np.zeros((H, W), dtype=np.uint8)
    mask_mismatch[50:60, 50:60] = 1
    dice_zero = calculate_dice(mask, mask_mismatch)
    assert dice_zero == 0.0, "Non-overlapping masks should have Dice 0.0"
    print("Dice Calculation: Success")

    # 3. Hausdorff (3D)
    # Create simple 3D volumes (Depth=2, H=10, W=10)
    vol_a = np.zeros((2, 10, 10), dtype=np.uint8)
    vol_a[0, 2:5, 2:5] = 1

    vol_b = np.zeros((2, 10, 10), dtype=np.uint8)
    vol_b[0, 2:5, 2:5] = 1  # Perfect match

    hd_perfect = calculate_hausdorff(vol_a, vol_b)
    assert hd_perfect == 0.0, f"Perfect Hausdorff should be 0.0, got {hd_perfect}"
    print("Hausdorff Calculation: Success")

    # 4. Largest Component
    noisy_mask = np.zeros((50, 50), dtype=np.uint8)
    noisy_mask[10:20, 10:20] = 1  # Large component (100 pixels)
    noisy_mask[40:42, 40:42] = 1  # Small component (4 pixels)

    cleaned = keep_largest_component(noisy_mask)
    assert cleaned[10, 10] == 1, "Large component should be kept"
    assert cleaned[40, 40] == 0, "Small component should be removed"
    print("Keep Largest Component: Success")


def demo_dataset_and_loader():
    print("\n=== Demonstrating Dataset and DataLoader ===")

    # Create a dummy subset of metadata to speed up processing
    full_meta_path = Config.TRAIN_METADATA_PATH
    if not os.path.exists(full_meta_path):
        raise FileNotFoundError(f"Metadata file not found: {full_meta_path}")

    df_full = pd.read_csv(full_meta_path)

    # Select a single case to ensure we have contiguous slices for 2.5D logic
    # We take the first available case
    case_id = df_full["case"].iloc[0]
    case_df = df_full[df_full["case"] == case_id]

    # Select the first 10 unique slice IDs to ensure we get exactly 10 processed samples
    target_ids = case_df["id"].unique()[:10]
    df_subset = case_df[case_df["id"].isin(target_ids)].copy()

    # Save to working dir
    subset_csv_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    df_subset.to_csv(subset_csv_path, index=False)

    # Run prepare_data on subset
    # We disable loading cached data to force processing our subset
    df_processed = prepare_data(subset_csv_path, mode="train", load_cached_data=False)

    assert "file_path_prev" in df_processed.columns
    assert "file_path_next" in df_processed.columns
    assert len(df_processed) == 10
    print("Data Preparation: Success")

    # Initialize Dataset
    transforms = get_transforms("train")
    dataset = GIDataset(df_processed, mode="train", transforms=transforms)

    # Fetch one sample
    sample = dataset[0]
    img = sample["image"]
    mask = sample["mask"]

    # Check shapes
    # Image: (3, H, W) -> 3 channels for 2.5D
    # Mask: (3, H, W) -> 3 classes
    # Due to RandomCrop in transforms, H and W should be Config.PATCH_SIZE
    expected_h, expected_w = Config.PATCH_SIZE

    assert img.shape == (
        3,
        expected_h,
        expected_w,
    ), f"Image shape mismatch: {img.shape}"
    assert mask.shape == (
        3,
        expected_h,
        expected_w,
    ), f"Mask shape mismatch: {mask.shape}"
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    print(f"Dataset Item Shape: Image {img.shape}, Mask {mask.shape}")
    print("Dataset & Transforms: Success")

    return dataset


def demo_model_and_training_step(dataset):
    print("\n=== Demonstrating Model, Loss, and Training Step ===")

    device = Config.DEVICE

    # Initialize Model
    model = UNetPlusPlus25D().to(device)

    # Create a small dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Get a batch
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    print(f"Batch Shapes: Images {images.shape}, Masks {masks.shape}")

    # Forward Pass
    outputs = model(images)

    # Check Deep Supervision Output
    if Config.DEEP_SUPERVISION:
        assert isinstance(
            outputs, list
        ), "Model should return list with deep supervision"
        print(f"Deep Supervision: Model returned {len(outputs)} outputs")
        main_output = outputs[0]
    else:
        main_output = outputs

    assert (
        main_output.shape == masks.shape
    ), f"Output shape {main_output.shape} mismatch with mask {masks.shape}"

    # Loss Calculation
    loss_fn = BCETverskyLoss()
    loss = loss_fn(outputs, masks)

    assert not torch.isnan(loss), "Loss is NaN"
    print(f"Calculated Loss: {loss.item():.4f}")

    # Backward Pass
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Backward Pass & Optimizer Step: Success")
    return model


def demo_inference(model):
    print("\n=== Demonstrating Sliding Window Inference ===")

    device = Config.DEVICE

    # Create a dummy image larger than the patch size to trigger sliding window
    # Shape: (3, 400, 400). Patch is (320, 320).
    H_orig, W_orig = 400, 400
    dummy_image = torch.randn(3, H_orig, W_orig)  # 2.5D input

    # Run inference
    # Note: predict_sliding_window expects the tensor to be on the correct device inside the function
    # but we pass the tensor. The function moves it to device.
    pred_map = predict_sliding_window(model, dummy_image, device)

    # Output should be numpy array (Num_Classes, H, W)
    assert isinstance(pred_map, np.ndarray)
    assert pred_map.shape == (
        Config.NUM_CLASSES,
        H_orig,
        W_orig,
    ), f"Prediction shape {pred_map.shape} mismatch. Expected {(Config.NUM_CLASSES, H_orig, W_orig)}"

    print(f"Inference Output Shape: {pred_map.shape}")
    print("Sliding Window Inference: Success")


def main():
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Utils
    demo_utils()

    # 2. Dataset
    dataset = demo_dataset_and_loader()

    # 3. Model & Training
    trained_model = demo_model_and_training_step(dataset)

    # 4. Inference
    demo_inference(trained_model)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
