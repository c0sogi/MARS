import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode, calc_map
from library.dataset import SaltDataset, load_data, get_transforms
from library.model import SaltModel
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.train import run_fold


def demo_rle_encoding():
    """Verifies RLE encoding and decoding logic."""
    print("\n--- Demo: RLE Encoding/Decoding ---")
    # Create a simple 10x10 mask with a 2x2 square of 1s
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = (
        1  # Pixels at (1,1), (1,2), (2,1), (2,2) -> Indices 11, 21, 12, 22 in column-major?
    )
    # Note: The problem statement says pixels are numbered top to bottom, then left to right.
    # (1,1) is 1. (2,1) is 2. (1,2) is 11 (if height is 10).

    # Let's use the provided functions to check consistency
    rle = rle_encode(mask)
    decoded_mask = rle_decode(rle, shape=(10, 10))

    assert np.array_equal(mask, decoded_mask), "RLE Round-trip failed"
    print("RLE Encode/Decode round-trip successful.")


def demo_metrics():
    """Verifies Mean Average Precision calculation."""
    print("\n--- Demo: Metric (mAP) ---")
    # Case 1: Perfect match
    pred = np.ones((1, 101, 101), dtype=np.float32)
    targ = np.ones((1, 101, 101), dtype=np.uint8)
    score = calc_map(pred, targ, pixel_threshold=0.5)
    assert np.isclose(score, 1.0), f"Expected mAP 1.0 for perfect match, got {score}"

    # Case 2: No overlap
    pred = np.zeros((1, 101, 101), dtype=np.float32)
    targ = np.ones((1, 101, 101), dtype=np.uint8)
    score = calc_map(pred, targ, pixel_threshold=0.5)
    assert np.isclose(score, 0.0), f"Expected mAP 0.0 for no overlap, got {score}"

    print("Metric calculation verified.")


def demo_dataset_and_model():
    """Verifies Dataset loading and Model forward pass."""
    print("\n--- Demo: Dataset and Model ---")

    # 1. Prepare a small subset of data
    full_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    subset_df = full_df.head(10).copy()

    # Use a temp cache dir for this demo to avoid conflicts
    temp_cache_dir = os.path.join(Config.WORKING_DIR, "demo_cache_unit_test")
    Config.CACHE_DIR = temp_cache_dir
    os.makedirs(temp_cache_dir, exist_ok=True)

    # 2. Load Data
    ids, images, masks, depths = load_data(
        subset_df, cache_name="demo_unit", load_cached_data=False
    )

    # 3. Instantiate Dataset
    ds = SaltDataset(images, depths, masks, transform=get_transforms("train"))

    # 4. Check Item
    img_tensor, mask_tensor = ds[0]
    print(f"Dataset Item Shapes - Image: {img_tensor.shape}, Mask: {mask_tensor.shape}")

    # Assertions
    # Image should be (3, 128, 128) due to padding and channel stacking
    assert img_tensor.shape == (
        3,
        128,
        128,
    ), f"Unexpected image shape: {img_tensor.shape}"
    assert mask_tensor.shape == (
        128,
        128,
    ), f"Unexpected mask shape: {mask_tensor.shape}"
    assert (
        img_tensor.max() <= 1.0 and img_tensor.min() >= 0.0
    ), "Image normalization failed"

    # 5. Model Forward Pass
    device = Config.DEVICE
    model = SaltModel(encoder_name="resnet18", pretrained=False, in_channels=3).to(
        device
    )
    model.eval()

    input_batch = img_tensor.unsqueeze(0).to(device)  # (1, 3, 128, 128)

    with torch.no_grad():
        output = model(input_batch)

    print(f"Model Output Shape (Eval): {output.shape}")
    assert output.shape == (1, 1, 128, 128), f"Unexpected output shape: {output.shape}"

    # 6. Loss Check
    # Dummy logits and targets
    logits = torch.randn(2, 1, 128, 128)
    targets = torch.randint(0, 2, (2, 128, 128)).float()

    loss_fn_1 = BCEDiceLoss()
    loss_1 = loss_fn_1(logits, targets)
    assert loss_1.item() > 0, "BCEDiceLoss should be positive"

    loss_fn_2 = LovaszHingeLoss()
    loss_2 = loss_fn_2(logits, targets)
    # Lovasz hinge can be 0 if perfect, but unlikely with random
    print(
        f"Loss Checks Passed. BCEDice: {loss_1.item():.4f}, Lovasz: {loss_2.item():.4f}"
    )

    # Cleanup unit test cache
    shutil.rmtree(temp_cache_dir, ignore_errors=True)


def run_integration_test():
    """Runs a full training fold on a reduced dataset."""
    print("\n--- Demo: Integration Test (Train Fold) ---")

    # 1. Setup Temporary Environment
    demo_working_dir = "./working/demo_execution"
    os.makedirs(demo_working_dir, exist_ok=True)

    # 2. Create Subset Metadata
    # We need enough samples for StratifiedKFold (n_splits=2) to work.
    # We take 50 samples to ensure class coverage.
    full_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    subset_df = full_df.head(50).copy()

    temp_meta_path = os.path.join(demo_working_dir, "temp_train_metadata.csv")
    subset_df.to_csv(temp_meta_path, index=False)

    # 3. Monkey-patch Config for Speed
    Config.WORKING_DIR = demo_working_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_working_dir, "checkpoints")
    Config.CACHE_DIR = os.path.join(demo_working_dir, "cache")
    Config.TRAIN_METADATA_PATH = temp_meta_path

    # Reduce compute load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.N_FOLDS = 2
    Config.ENCODER = "resnet18"  # Smaller encoder for speed
    Config.LOVASZ_SWITCH_EPOCH = 1  # Switch to Lovasz quickly to test both losses

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 4. Run Fold
    try:
        best_map = run_fold(fold_idx=0, debug=False)
        print(f"Integration test completed. Best mAP: {best_map:.4f}")
        assert best_map >= 0.0, "mAP should be non-negative"
    except Exception as e:
        print(f"Integration test failed with error: {e}")
        raise e
    finally:
        # Optional: Cleanup
        # shutil.rmtree(demo_working_dir, ignore_errors=True)
        pass


if __name__ == "__main__":
    # Initialize Config (sets up directories and DEVICE)
    Config.setup()

    # Ensure reproducibility
    seed_everything(42)

    # Run Demos
    demo_rle_encoding()
    demo_metrics()
    demo_dataset_and_model()
    run_integration_test()

    print("\nAll demonstrations completed successfully.")
