import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
import warnings
from unittest.mock import MagicMock

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, get_tissue_mask
from library.dataset import HubmapDataset
from library.model import UnetPlusPlus
from library.losses import DeepSupervisionLoss
from library.train import train_model
from library.inference import make_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("--- Starting HuBMAP Pipeline Demonstration ---")

    # 1. Setup & Configuration Overrides
    # We use a specific subdirectory in working to avoid clutter
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters for speed/demo purposes
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce compute requirements
    Config.TILE_SIZE = 512  # Smaller tiles for speed
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for demo run.")

    # 2. Data Optimization (Mini Metadata)
    # Create subsets of metadata to speed up dataset initialization (cache generation)
    print("\n--- Preparing Mini Datasets ---")

    train_meta_path = "./metadata/train.csv"
    val_meta_path = "./metadata/val.csv"
    test_meta_path = "./metadata/test.csv"

    # Read original metadata
    df_train = pd.read_csv(train_meta_path)
    df_val = pd.read_csv(val_meta_path)
    df_test = pd.read_csv(test_meta_path)

    # Save mini versions (1 sample each)
    mini_train_path = os.path.join(DEMO_DIR, "train_mini.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val_mini.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test_mini.csv")

    df_train.head(1).to_csv(mini_train_path, index=False)
    df_val.head(1).to_csv(mini_val_path, index=False)
    df_test.head(1).to_csv(mini_test_path, index=False)

    # Point Config to mini metadata
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"Mini metadata created at {DEMO_DIR}")

    # 3. Verify Utility Functions
    print("\n--- Verifying Utilities ---")

    # Test RLE Encode/Decode
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1
    rle = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle, (100, 100))

    assert isinstance(rle, str), "RLE should be a string"
    assert np.array_equal(dummy_mask, decoded_mask), "RLE Decode mismatch"
    print("RLE Encode/Decode verified.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")

    model = UnetPlusPlus(
        backbone_name=Config.BACKBONE,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )
    model.eval()

    # Create dummy input tensor (B, C, H, W)
    dummy_input = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE)

    with torch.no_grad():
        outputs = model(dummy_input)

    # UnetPlusPlus with Deep Supervision returns a list of tensors
    assert isinstance(outputs, list), "Model output should be a list (Deep Supervision)"
    assert (
        len(outputs) == 4
    ), f"Expected 4 outputs from Deep Supervision, got {len(outputs)}"
    assert outputs[0].shape == (
        2,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Output shape mismatch. Expected (2, 1, {Config.TILE_SIZE}, {Config.TILE_SIZE}), got {outputs[0].shape}"

    print("Model forward pass verified.")

    # 5. Verify Loss Function
    print("\n--- Verifying Loss Function ---")

    criterion = DeepSupervisionLoss(weights=Config.LOSS_WEIGHTS)
    dummy_target = torch.randint(
        0, 2, (2, 1, Config.TILE_SIZE, Config.TILE_SIZE)
    ).float()

    # Calculate loss
    loss = criterion(outputs, dummy_target)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"Loss calculation verified. Value: {loss.item():.4f}")

    # 6. Verify Dataset Loading
    print("\n--- Verifying Dataset ---")

    # Instantiate dataset (this will trigger cache generation for the mini dataset)
    # Note: We pass cache_dir explicitly to ensure it uses our demo dir
    dataset = HubmapDataset(
        mode="train", samples_per_epoch=4, cache_dir=Config.CACHE_DIR
    )

    assert len(dataset) == 4, "Dataset length mismatch"

    # Fetch one item
    img, mask = dataset[0]

    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert isinstance(mask, torch.Tensor), "Mask should be a tensor"
    assert img.shape == (
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Image shape mismatch: {img.shape}"
    assert mask.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Mask shape mismatch: {mask.shape}"

    print("Dataset loading and caching verified.")

    # 7. Demonstrate Training Loop
    print("\n--- Demonstrating Training Loop ---")

    # We call train_model with limited scope
    # train_model internally instantiates datasets. It uses Config.CACHE_DIR default which is set at import time.
    # However, since we modified Config attributes, the internal logic using Config.ATTR works.
    # The only catch is the default arg for HubmapDataset inside train.py.
    # Since we can't easily change the default arg of a function in another module after import without complex patching,
    # we rely on the fact that HubmapDataset uses Config.CACHE_DIR if passed, or we accept it might use the default.
    # To be safe and ensure it works with our mini-data, we will patch HubmapDataset in library.train temporarily
    # or just trust the Config update.
    # Actually, HubmapDataset signature is `def __init__(..., cache_dir=Config.CACHE_DIR)`.
    # The default was evaluated at import.
    # Let's verify if `train_model` passes cache_dir. It does NOT.
    # So `HubmapDataset` inside `train.py` will use the OLD `Config.CACHE_DIR` ("./working/idea_6/cache").
    # This is fine, as long as it writes there. But we prepared data in `./working/demo_run/cache`.
    # FIX: We will monkey-patch HubmapDataset in library.train to force our cache dir.

    import library.train

    original_dataset_cls = library.train.HubmapDataset

    class PatchedHubmapDataset(HubmapDataset):
        def __init__(self, *args, **kwargs):
            # Force cache_dir to our demo dir
            kwargs["cache_dir"] = Config.CACHE_DIR
            super().__init__(*args, **kwargs)

    library.train.HubmapDataset = PatchedHubmapDataset

    try:
        trained_model = train_model(
            num_epochs=1, batch_size=2, samples_per_epoch=4, patience=1
        )
    finally:
        # Restore original class
        library.train.HubmapDataset = original_dataset_cls

    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created"
    print("Training loop completed successfully.")

    # 8. Demonstrate Inference
    print("\n--- Demonstrating Inference ---")

    # Inference on large images takes too long for a demo.
    # We will monkey-patch `get_tissue_mask` in `library.inference` to return a mask
    # that is valid ONLY for a tiny region (top-left).
    # This forces the sliding window to skip 99% of the image, making inference instant.

    import library.inference

    # Define the mock function
    def mocked_get_tissue_mask(
        image_id, width, height, anatomical_json_path, load_cached_data=True
    ):
        # Create a mask that is all zeros except for a small 512x512 box at 0,0
        # This ensures the sliding window only processes the first tile.
        mask = np.zeros((height, width), dtype=np.uint8)
        active_h = min(height, Config.TILE_SIZE)
        active_w = min(width, Config.TILE_SIZE)
        mask[0:active_h, 0:active_w] = 1
        return mask

    # Apply the patch
    original_get_mask_inference = library.inference.get_tissue_mask
    library.inference.get_tissue_mask = mocked_get_tissue_mask

    try:
        make_submission(
            checkpoint_path=Config.CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
        )
    finally:
        # Restore original function
        library.inference.get_tissue_mask = original_get_mask_inference

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())
    assert len(df_sub) > 0, "Submission file is empty"

    print("\n--- Demonstration Complete ---")
    print(f"Artifacts stored in {DEMO_DIR}")


if __name__ == "__main__":
    main()
