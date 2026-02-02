import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

# Import from the provided library
from library.utils import (
    set_seed,
    load_dicom_slice,
    read_dicom_slab,
    independent_slab_normalize,
    get_brain_depth_range,
)
from library.data import TMSVDataset
from library.model import TMSVNet
from library.config import IMAGE_SIZE, DEVICE


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Reproducibility
    set_seed(42)
    print("Seed set to 42.")

    # Define a sample subject path for demonstration
    # Using Subject 00000 from the training set
    subject_id = "00000"
    base_path = os.path.join("./input/train", subject_id)

    # Paths for the 3 streams
    flair_path = os.path.join(base_path, "FLAIR")
    t1wce_path = os.path.join(base_path, "T1wCE")
    t2w_path = os.path.join(base_path, "T2w")

    # ==========================================
    # 2. Demonstrate library.utils
    # ==========================================
    print("\n--- Testing library.utils ---")

    # Test load_dicom_slice
    # We pick an arbitrary file we know exists in the structure
    sample_file = os.path.join(flair_path, "Image-100.dcm")
    if os.path.exists(sample_file):
        img = load_dicom_slice(sample_file, target_size=IMAGE_SIZE)
        assert img is not None, "Failed to load DICOM slice"
        assert img.shape == IMAGE_SIZE, f"Expected shape {IMAGE_SIZE}, got {img.shape}"
        print(f"load_dicom_slice: Successfully loaded image with shape {img.shape}")
    else:
        print(
            f"Warning: Sample file {sample_file} not found. Skipping single slice test."
        )

    # Test get_brain_depth_range
    # This scans the directory to find where the brain starts and ends
    min_idx, max_idx = get_brain_depth_range(flair_path)
    print(f"get_brain_depth_range: Detected brain range [{min_idx}, {max_idx}]")
    assert max_idx >= min_idx, "Invalid brain depth range detected"

    # Test read_dicom_slab and independent_slab_normalize
    # We'll use the middle of the brain as the center index
    center_idx = (min_idx + max_idx) // 2

    # Load slabs for all 3 modalities (needed for the model later)
    # Shape should be (H, W, 3)
    slab_flair = read_dicom_slab(flair_path, center_idx, slab_depth=3)
    slab_t1wce = read_dicom_slab(t1wce_path, center_idx, slab_depth=3)
    slab_t2w = read_dicom_slab(t2w_path, center_idx, slab_depth=3)

    assert slab_flair.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), "Incorrect slab shape"
    print(f"read_dicom_slab: Loaded slab with shape {slab_flair.shape}")

    # Normalize
    norm_flair = independent_slab_normalize(slab_flair)
    norm_t1wce = independent_slab_normalize(slab_t1wce)
    norm_t2w = independent_slab_normalize(slab_t2w)

    assert norm_flair.max() <= 1.0 + 1e-6, "Normalization failed (max > 1)"
    assert norm_flair.min() >= 0.0 - 1e-6, "Normalization failed (min < 0)"
    print("independent_slab_normalize: Normalization verified.")

    # ==========================================
    # 3. Demonstrate library.data.TMSVDataset
    # ==========================================
    print("\n--- Testing library.data.TMSVDataset ---")

    # Construct a mini-dataset manually to avoid processing the whole input folder
    # We will duplicate the loaded slabs to create a batch of size 4
    batch_size = 4

    flair_data = np.array([norm_flair] * batch_size, dtype=np.float32)
    t1wce_data = np.array([norm_t1wce] * batch_size, dtype=np.float32)
    t2w_data = np.array([norm_t2w] * batch_size, dtype=np.float32)

    # Dummy targets (0 or 1) and IDs
    targets = np.array([0, 1, 0, 1], dtype=np.float32)
    ids = np.array([int(subject_id)] * batch_size, dtype=np.int64)

    # Instantiate Dataset
    dataset = TMSVDataset(
        flair_data=flair_data,
        t1wce_data=t1wce_data,
        t2w_data=t2w_data,
        targets=targets,
        ids=ids,
        transform=None,  # Skipping augmentation for deterministic demo
    )

    # Validate __getitem__
    sample = dataset[0]
    required_keys = ["flair", "t1wce", "t2w", "BraTS21ID", "target"]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset sample"

    # Check tensor shapes: (Channels, H, W) -> (3, 256, 256)
    expected_tensor_shape = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    assert (
        sample["flair"].shape == expected_tensor_shape
    ), f"Expected tensor shape {expected_tensor_shape}, got {sample['flair'].shape}"
    assert isinstance(sample["flair"], torch.Tensor), "Output is not a torch Tensor"

    print("TMSVDataset: Successfully created dataset and verified item structure.")

    # ==========================================
    # 4. Demonstrate library.model.TMSVNet
    # ==========================================
    print("\n--- Testing library.model.TMSVNet ---")

    # Initialize Model
    # Using 'efficientnet_b0' as per config, pretrained=True
    model = TMSVNet(
        pretrained=False
    )  # False for speed in demo (no download needed if cached, but safer)
    model.to(DEVICE)
    model.train()  # Set to train mode

    print("TMSVNet: Model instantiated successfully.")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Get a batch
    batch = next(iter(dataloader))

    # Move to device
    b_flair = batch["flair"].to(DEVICE)
    b_t1wce = batch["t1wce"].to(DEVICE)
    b_t2w = batch["t2w"].to(DEVICE)
    b_targets = batch["target"].to(DEVICE)

    # Forward Pass
    logits = model(b_flair, b_t1wce, b_t2w)

    # Check Output Shape: (Batch_Size, Num_Classes) -> (2, 1)
    assert logits.shape == (2, 1), f"Expected output shape (2, 1), got {logits.shape}"
    print(f"TMSVNet: Forward pass successful. Output shape: {logits.shape}")

    # ==========================================
    # 5. Demonstrate Training Step (library.train logic)
    # ==========================================
    print("\n--- Testing Training Step ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Zero gradients
    optimizer.zero_grad()

    # Calculate loss
    loss = criterion(logits, b_targets)

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    print(f"Training Step: Loss computed successfully: {loss.item():.6f}")
    assert not np.isnan(loss.item()), "Loss is NaN"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
