import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_encoding, fbeta_score, dice_coef
from library.dataset import InkDataset
from library.model import SpecialistModel
from library.trainer import train_specialist
from library.inference import predict_slab, fuse_predictions


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly to suit a quick demo run.
    print("\n[1] Configuring environment...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EXPERIMENT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("working", Config.EXPERIMENT_NAME)
    Config.VALID_THRESHOLD = -1.0  # Force checkpoint saving

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Utility Verification
    print("\n[2] Verifying Utilities...")

    # Test RLE Encoding
    # Mask: 0 1 1 0 0 1 0 -> Indices (1-based): 2,3 and 6. Runs: start 2 len 2, start 6 len 1
    dummy_mask = np.array([[0, 1, 1, 0], [0, 1, 0, 0]])
    # Flattened: 0 1 1 0 0 1 0 0
    # Runs: 2 2, 6 1
    rle_out = rle_encoding(dummy_mask)
    expected_rle = "2 2 6 1"
    assert (
        rle_out == expected_rle
    ), f"RLE Encoding failed. Got {rle_out}, expected {expected_rle}"
    print("  RLE Encoding: OK")

    # Test F-Beta Score
    # Preds: 0.8 (TP), 0.2 (TN), 0.8 (FP), 0.2 (FN) -> with threshold 0.5
    # Targets: 1, 0, 0, 1
    dummy_preds = torch.tensor([0.8, 0.2, 0.8, 0.2])
    dummy_targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
    # TP=1, FP=1, FN=1. Precision=0.5, Recall=0.5. F0.5 should be calculated.
    score = fbeta_score(dummy_preds, dummy_targets, beta=0.5, threshold=0.5)
    assert 0.0 <= score <= 1.0, "F-Beta score out of range"
    print(f"  F-Beta Score: {score:.4f} (OK)")

    # 3. Dataset Demonstration
    print("\n[3] Demonstrating Dataset Loading...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA}")

    df_train = pd.read_csv(Config.TRAIN_METADATA)

    # Filter to a tiny subset to speed up volume loading (just 4 samples from one fragment)
    # We pick fragment '1' as it is generally available in the sample data
    subset_df = df_train[df_train["fragment_id"].astype(str) == "1"].head(4).copy()
    if len(subset_df) == 0:
        # Fallback if fragment 1 not found, just take head
        subset_df = df_train.head(4).copy()

    print(f"  Selected {len(subset_df)} samples for demonstration.")

    # Instantiate Dataset for the 'mid' specialist range (Z: 20-44)
    z_start, z_end = 20, 44
    dataset = InkDataset(
        metadata=subset_df,
        z_start=z_start,
        z_end=z_end,
        mode="train",
        load_cached_data=True,
    )

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Fetch one batch
    batch = next(iter(loader))
    images = batch["image"]
    labels = batch["label"]
    valid_masks = batch["valid_mask"]

    # Verify Shapes
    # Image: (B, 3, 512, 512) - 3 channels because of the slab projection
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Unexpected image shape: {images.shape}"
    # Label: (B, 1, 512, 512)
    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Unexpected label shape: {labels.shape}"

    print(f"  Batch Shapes Verified: Image {images.shape}, Label {labels.shape}")
    print("  Dataset Loading: OK")

    # 4. Model Demonstration
    print("\n[4] Demonstrating Model Initialization & Forward Pass...")

    device = Config.DEVICE
    model = SpecialistModel(model_name=Config.BACKBONE)
    model.to(device)

    # Forward pass
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    # Verify Output Shape: (B, 1, 512, 512)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Unexpected output shape: {outputs.shape}"

    print(f"  Model Output Shape: {outputs.shape}")
    print("  Model Forward Pass: OK")

    # 5. Training Loop Demonstration
    print("\n[5] Demonstrating Training Loop...")

    # Define a specialist config
    specialist_config = {
        "name": "demo_mid",
        "z_start": z_start,
        "z_end": z_end,
        "checkpoint_path": os.path.join(Config.WORKING_DIR, "best_model.pth"),
    }

    # Use the same loader for train and val to save time
    best_score = train_specialist(model, loader, loader, specialist_config)

    print(f"  Training complete. Best Val F0.5: {best_score:.4f}")
    assert os.path.exists(
        specialist_config["checkpoint_path"]
    ), "Checkpoint file was not created."
    print("  Checkpoint saved successfully.")

    # 6. Inference Component Demonstration
    print("\n[6] Demonstrating Inference Components...")

    # We will simulate inference using the dataset we already have (subset_df)
    # acting as a "test" set.

    # Switch dataset to test mode (no labels returned, different transforms)
    test_dataset = InkDataset(
        metadata=subset_df,
        z_start=z_start,
        z_end=z_end,
        mode="test",
        load_cached_data=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Define the shape of the full fragment for reconstruction
    # We get this from the mask associated with the first sample
    sample_row = subset_df.iloc[0]
    mask_path = os.path.join(Config.INPUT_DIR, sample_row["mask_path"])
    full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    full_shape = full_mask.shape

    print(f"  Simulating inference on fragment shape: {full_shape}")

    # Run predict_slab
    # This predicts on tiles and stitches them into the full_shape
    prob_map = predict_slab(model, test_loader, full_shape, device)

    assert (
        prob_map.shape == full_shape
    ), f"Probability map shape mismatch. Got {prob_map.shape}"
    assert prob_map.dtype == np.float32, "Probability map should be float32"
    print("  predict_slab: OK")

    # Test Fuse Predictions (Max Fusion)
    # Simulate having 2 predictions (e.g., from different specialists)
    map1 = np.random.rand(*full_shape).astype(np.float32)
    map2 = np.random.rand(*full_shape).astype(np.float32)

    fused = fuse_predictions([map1, map2])

    # Check max logic at a specific point
    y, x = 0, 0
    expected_val = max(map1[y, x], map2[y, x])
    assert np.isclose(fused[y, x], expected_val), "Fusion logic incorrect"
    print("  fuse_predictions: OK")

    # Final Encoding Check on the generated map
    binary_map = (prob_map > 0.5).astype(np.uint8)
    final_rle = rle_encoding(binary_map)
    print(f"  Final RLE length: {len(final_rle)} chars")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
