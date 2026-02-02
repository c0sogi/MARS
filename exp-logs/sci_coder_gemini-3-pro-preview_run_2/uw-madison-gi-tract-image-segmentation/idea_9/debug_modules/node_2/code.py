import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import cv2
import warnings

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, rle_decode, keep_largest_component_3d, dice_coef
from library.dataset import UWMDataset, get_loaders
from library.model import BiSeNet25D
from library.loss import BCEDiceLoss
from library.train import train_one_epoch, validate
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """Sets up a temporary environment for the demo."""
    # Define demo paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config to use demo directory and settings
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.CACHE_PATH = os.path.join(demo_dir, "data_cache.parquet")

    # Speed up execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Very small subset for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Setup standard seeds
    Config.setup(seed=42)

    print(f"Demo environment set up at: {demo_dir}")
    return demo_dir


def verify_utils():
    """Verifies utility functions: RLE and 3D processing."""
    print("\n=== Verifying Utilities ===")

    # 1. RLE Encode/Decode
    # Create a simple 10x10 mask with a 2x2 square of 1s
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, (10, 10))

    assert np.array_equal(mask, decoded), "RLE Decode does not match original mask"
    print("RLE Encode/Decode verification passed.")

    # 2. 3D Largest Component
    # Create a 3D volume (3 slices, 10x10)
    # Slice 0: Small dot (noise)
    # Slice 1: Large square (target)
    # Slice 2: Large square (target)
    vol = np.zeros((3, 10, 10), dtype=np.uint8)
    vol[0, 8, 8] = 1  # Noise
    vol[1, 2:6, 2:6] = 1  # Component 1 (Size 16)
    vol[2, 2:6, 2:6] = 1  # Component 1 (Size 16, connected to slice 1)

    # The noise is unconnected to the main block.
    # Total size of main block = 32. Noise size = 1.

    cleaned_vol = keep_largest_component_3d(vol)

    assert cleaned_vol[0, 8, 8] == 0, "Noise should be removed"
    assert cleaned_vol[1, 2, 2] == 1, "Main component should remain"
    print("3D Largest Component verification passed.")


def verify_dataset_and_loader():
    """Verifies Dataset creation and DataLoader functionality."""
    print("\n=== Verifying Dataset & DataLoader ===")

    # Load real metadata
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Sample a subset for the demo to ensure 2.5D logic works
    # We pick a specific case to ensure continuity
    case_id = full_train_df["case"].iloc[0]
    subset_df = (
        full_train_df[full_train_df["case"] == case_id].head(30).reset_index(drop=True)
    )

    print(f"Created subset with {len(subset_df)} rows from Case {case_id}")

    # Initialize Dataset
    # Note: load_cached_data=False forces processing the subset immediately
    ds = UWMDataset(subset_df, phase="train", load_cached_data=False)

    assert len(ds) > 0, "Dataset should not be empty"

    # Check item
    img, mask, img_id = ds[10]  # Pick a middle index to ensure neighbors exist

    # Check shapes
    # Image: (3, H, W) -> 3 channels for 2.5D
    # Mask: (3, H, W) -> 3 classes
    assert img.shape == (
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Image shape mismatch: {img.shape}"
    assert mask.shape == (
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Mask shape mismatch: {mask.shape}"
    assert isinstance(img, torch.Tensor), "Image should be a tensor"

    print(f"Dataset item shape verified: Img {img.shape}, Mask {mask.shape}")

    # Check Loader
    train_loader, _, _ = get_loaders(
        subset_df,
        subset_df,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        load_cached_data=False,
    )
    batch_img, batch_mask, batch_ids = next(iter(train_loader))

    assert batch_img.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    print("DataLoader batch retrieval verified.")

    return train_loader


def verify_model_and_training(train_loader):
    """Verifies Model initialization, Forward pass, Loss, and Training Loop."""
    print("\n=== Verifying Model & Training ===")

    device = Config.DEVICE

    # 1. Initialize Model
    model = BiSeNet25D(num_classes=Config.NUM_CLASSES).to(device)
    print(f"Model {Config.MODEL_NAME} initialized.")

    # 2. Forward Pass Check
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    main_out, aux_out = model(dummy_input)

    assert main_out.shape == (
        2,
        3,
        256,
        256,
    ), f"Main output shape mismatch: {main_out.shape}"
    assert aux_out.shape == (
        2,
        3,
        256,
        256,
    ), f"Aux output shape mismatch: {aux_out.shape}"
    print("Model forward pass verified.")

    # 3. Loss Check
    criterion = BCEDiceLoss()
    dummy_target = torch.randint(0, 2, (2, 3, 256, 256)).float().to(device)

    loss = criterion((main_out, aux_out), dummy_target)
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"Loss calculation verified: {loss.item():.4f}")

    # 4. Training Loop Demo
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print("Running 1 epoch of training on subset...")
    # Using the train_loader from previous step
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Training epoch complete. Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_dice = validate(model, train_loader, criterion, device)
    print(f"Validation complete. Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

    # Save model for inference step
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")


def verify_inference():
    """Verifies the inference pipeline."""
    print("\n=== Verifying Inference ===")

    # 1. Create a mini test metadata file
    # We will reuse the subset from training but pretend it's test data
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Take a small slice of a case
    test_subset = full_train_df.head(20).copy()

    # Test metadata format requires 'predicted' column (can be empty)
    # and usually doesn't have 'segmentation' (though our code handles it via 'class' check)
    test_subset["predicted"] = ""

    # Save to the location Config expects (we overrode Config.TEST_METADATA_PATH logic?
    # No, Config defines paths based on METADATA_DIR. We need to override the path in Config class or file)

    # Since Config is a class with static attributes, we can patch it.
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test_metadata.csv")
    test_subset.to_csv(mini_test_path, index=False)
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"Created mini test metadata at {mini_test_path}")

    # 2. Run Prediction
    # We use the predict_and_submit function which encapsulates loading, inference, 3D proc, and saving
    # We pass load_cached_data=False to force it to read our new mini file
    # We pass debug=True (though we already set Config.DEBUG)

    try:
        predict_and_submit(load_cached_data=False, debug=True)
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # 3. Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated successfully. Rows: {len(sub_df)}")
        print(sub_df.head())

        # Basic check
        assert "id" in sub_df.columns
        assert "class" in sub_df.columns
        assert "predicted" in sub_df.columns
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    print("Starting Demo Script...")

    # 1. Setup
    setup_demo_environment()

    # 2. Utils
    verify_utils()

    # 3. Data
    loader = verify_dataset_and_loader()

    # 4. Model & Train
    verify_model_and_training(loader)

    # 5. Inference
    verify_inference()

    print("\nAll demonstrations completed successfully.")
