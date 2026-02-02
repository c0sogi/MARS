import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import library modules
import library.config as config
import library.utils as utils
from library.data_interface import DataInterface
from library.preprocessing import load_lidar_points, points_to_bev, generate_target_maps
from library.dataset import BEVDataset
from library.model import BEVDetector
from library.loss import BEVLoss
from library.train import train_model
from library.inference import generate_submission


def run_demo():
    print("=== Starting 3D Object Detection Pipeline Demo ===")

    # 1. Setup & Configuration Override
    # We create small subsets of the metadata to ensure the demo runs quickly.
    subset_size = 16
    working_dir = config.WORKING_DIR
    os.makedirs(working_dir, exist_ok=True)

    print(f"Creating metadata subsets (size={subset_size})...")

    # Create Train Subset
    train_df = pd.read_csv(config.TRAIN_METADATA)
    train_subset_path = os.path.join(working_dir, "train_subset.csv")
    train_df.head(subset_size).to_csv(train_subset_path, index=False)

    # Create Val Subset
    val_df = pd.read_csv(config.VAL_METADATA)
    val_subset_path = os.path.join(working_dir, "val_subset.csv")
    val_df.head(subset_size).to_csv(val_subset_path, index=False)

    # Create Test Subset
    test_df = pd.read_csv(config.TEST_METADATA)
    test_subset_path = os.path.join(working_dir, "test_subset.csv")
    test_df.head(subset_size).to_csv(test_subset_path, index=False)

    # Monkey-patch config to use these subsets and reduce load
    config.TRAIN_METADATA = train_subset_path
    config.VAL_METADATA = val_subset_path
    config.TEST_METADATA = test_subset_path
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    config.set_seed(42)

    # 2. Validate Data Interface
    print("\n--- Validating DataInterface ---")
    data_interface = DataInterface(
        load_cached_data=False
    )  # Force recompute to test logic

    # Test get_transform_matrix
    sample_token = train_df.iloc[0]["sample_token"]
    transform_matrix = data_interface.get_transform_matrix(sample_token)

    print(f"Transform Matrix Shape: {transform_matrix.shape}")
    if transform_matrix.shape != (4, 4):
        raise AssertionError("Transform matrix must be 4x4")
    if not np.allclose(transform_matrix[3, :], [0, 0, 0, 1]):
        raise AssertionError("Transform matrix must be homogeneous")
    print("DataInterface validation successful.")

    # 3. Validate Preprocessing (LiDAR -> BEV)
    print("\n--- Validating Preprocessing ---")
    lidar_rel_path = train_df.iloc[0]["lidar_path"]
    # Fix path if it starts with ./
    if lidar_rel_path.startswith("./"):
        lidar_path = lidar_rel_path
    else:
        lidar_path = os.path.join(config.INPUT_DIR, lidar_rel_path)

    # Check if file exists (it should based on metadata)
    if not os.path.exists(lidar_path):
        print(
            f"Warning: LiDAR file not found at {lidar_path}. Using random points for demo."
        )
        points = np.random.rand(1000, 4).astype(np.float32) * 100
    else:
        points = load_lidar_points(lidar_path)

    print(f"Loaded LiDAR points: {points.shape}")

    bev_map = points_to_bev(points)
    print(f"Generated BEV Map Shape: {bev_map.shape}")

    expected_bev_shape = (config.BEV_CHANNELS, config.GRID_SIZE[1], config.GRID_SIZE[0])
    if bev_map.shape != expected_bev_shape:
        raise AssertionError(
            f"BEV shape mismatch. Expected {expected_bev_shape}, got {bev_map.shape}"
        )
    print("Preprocessing validation successful.")

    # 4. Validate Dataset
    print("\n--- Validating Dataset ---")
    train_dataset = BEVDataset(
        split="train", data_interface=data_interface, load_cached_data=False
    )
    sample = train_dataset[0]

    print("Dataset Sample Keys:", sample.keys())
    input_tensor = sample["input"]
    hm_target = sample["hm"]
    reg_target = sample["reg"]

    print(f"Input Tensor: {input_tensor.shape}")
    print(f"Heatmap Target: {hm_target.shape}")
    print(f"Regression Target: {reg_target.shape}")

    if input_tensor.shape != expected_bev_shape:
        raise AssertionError("Dataset input tensor shape mismatch")

    # Check downsampling of targets
    expected_h_out = config.GRID_SIZE[1] // config.DOWN_RATIO
    expected_w_out = config.GRID_SIZE[0] // config.DOWN_RATIO

    if hm_target.shape[1:] != (expected_h_out, expected_w_out):
        raise AssertionError(
            f"Target spatial dims mismatch. Expected {(expected_h_out, expected_w_out)}"
        )

    print("Dataset validation successful.")

    # 5. Validate Model & Loss
    print("\n--- Validating Model & Loss ---")
    device = config.get_device()
    model = BEVDetector().to(device)
    criterion = BEVLoss()

    # Prepare batch
    input_batch = input_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    targets_batch = {
        "hm": hm_target.unsqueeze(0).to(device),
        "reg": reg_target.unsqueeze(0).to(device),
        "reg_mask": sample["reg_mask"].unsqueeze(0).to(device),
    }

    # Forward
    hm_pred, reg_pred = model(input_batch)
    print(f"Model Output HM: {hm_pred.shape}")
    print(f"Model Output Reg: {reg_pred.shape}")

    if hm_pred.shape != targets_batch["hm"].shape:
        raise AssertionError("Model output shape does not match target shape")

    # Loss
    loss, stats = criterion((hm_pred, reg_pred), targets_batch)
    print(f"Calculated Loss: {loss.item()}")

    if torch.isnan(loss):
        raise AssertionError("Loss is NaN")
    print("Model & Loss validation successful.")

    # 6. Run Training Loop
    print("\n--- Running Training Loop (Subset) ---")
    # This uses the library.train.train_model function
    # It will use the monkey-patched config for paths and epochs
    trained_model = train_model(
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        load_cached_data=True,  # Use caching to speed up second access
        num_workers=config.NUM_WORKERS,
    )

    model_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        # If validation loss didn't improve (unlikely with 1 epoch if init is bad, but possible),
        # save the current model manually for inference testing
        torch.save(trained_model.state_dict(), model_path)
        print("Saved model manually for inference step.")
    else:
        print("Best model found and saved.")

    print("Training loop execution successful.")

    # 7. Run Inference
    print("\n--- Running Inference (Subset) ---")
    generate_submission(
        model_path=model_path,
        batch_size=config.BATCH_SIZE,
        load_cached_data=True,
        num_workers=config.NUM_WORKERS,
        threshold=0.1,
    )

    if not os.path.exists(config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created")

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")
    print("Head of submission:")
    print(sub_df.head())

    # Verify format
    if "Id" not in sub_df.columns or "PredictionString" not in sub_df.columns:
        raise AssertionError("Submission columns mismatch")

    print("Inference execution successful.")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    run_demo()
