import os
import sys
import pandas as pd
import numpy as np
import torch
import logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, read_dicom_robust
from library.data import DataCacher, MRIDataset, get_transforms
from library.model import AsymmetricEfficientNet
from library.train import train_model
from library.inference import Predictor

# Configure logging to be less verbose for libraries that might use it
logging.getLogger("library.data").setLevel(logging.WARNING)
logging.getLogger("library.train").setLevel(logging.INFO)
logging.getLogger("library.inference").setLevel(logging.INFO)


def run_demo():
    print("=== Starting Demonstration of Glioblastoma MGMT Pipeline ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG_DATA_LIMIT = 6  # Only process 6 subjects for train/val/test
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny datasets
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")

    # Ensure working directories exist
    Config.setup()
    seed_everything(Config.SEED)

    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Data Limit: {Config.DEBUG_DATA_LIMIT}")
    print("    Configuration complete.\n")

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions (DICOM Reading)
    # --------------------------------------------------------------------------
    print("[2] Verifying robust DICOM reader...")

    # Load metadata to get a valid file path
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    sample_row = df_train.iloc[0]

    # Construct a full path to a FLAIR image
    flair_dir = os.path.join(Config.INPUT_DIR, sample_row["path_FLAIR"])
    # Get first file in directory
    files = sorted(os.listdir(flair_dir))
    if files:
        sample_file = os.path.join(flair_dir, files[0])

        # Test reading
        img = read_dicom_robust(sample_file, target_size=Config.IMG_SIZE)

        # Assertions
        assert isinstance(img, np.ndarray), "Reader must return a numpy array"
        assert img.shape == (
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Expected shape ({Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
        assert img.dtype == np.float32, "Image must be float32"

        print(
            f"    Successfully read {os.path.basename(sample_file)} with shape {img.shape}"
        )
    else:
        print("    Warning: No files found in sample directory to test reader.")
    print("    DICOM reader verification passed.\n")

    # --------------------------------------------------------------------------
    # 3. Verify Data Pipeline (Cacher & Dataset)
    # --------------------------------------------------------------------------
    print("[3] Verifying DataCacher and MRIDataset...")

    # We use a small subset of metadata for this test
    df_subset = df_train.head(Config.DEBUG_DATA_LIMIT).copy()

    # Run Cacher
    # Note: This will save to the demo_cache directory defined in step 1
    cache_data = DataCacher.process_data(
        df_subset, cache_key="demo_train", load_cached_data=False
    )

    assert "images" in cache_data, "Cache must contain 'images' key"
    assert "roi_centers" in cache_data, "Cache must contain 'roi_centers' key"
    assert len(cache_data["images"]) == len(df_subset), "Cache size mismatch"

    # Initialize Dataset
    dataset = MRIDataset(
        data_cache=cache_data,
        metadata_df=df_subset,
        transform=get_transforms("train"),
        stride_mode="random",
    )

    # Fetch one item
    tensor_img, label = dataset[0]

    # Assertions
    # Expected shape: (Channels=12, H=224, W=224)
    expected_channels = Config.TOTAL_CHANNELS
    assert tensor_img.shape == (
        expected_channels,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Dataset tensor shape mismatch. Expected {(expected_channels, Config.IMG_SIZE, Config.IMG_SIZE)}, got {tensor_img.shape}"
    assert isinstance(label, torch.Tensor), "Label must be a torch Tensor"

    print(f"    Dataset yielded tensor: {tensor_img.shape} and label: {label.item()}")
    print("    Data pipeline verification passed.\n")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("[4] Verifying AsymmetricEfficientNet...")

    model = AsymmetricEfficientNet()
    model.eval()

    # Create dummy input batch: (Batch=2, Channels=12, H=224, W=224)
    dummy_input = torch.randn(
        2, Config.TOTAL_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE
    )

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    # Assertions
    assert logits.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {logits.shape}"

    print(
        f"    Model accepted input {dummy_input.shape} and produced logits {logits.shape}"
    )
    print("    Model verification passed.\n")

    # --------------------------------------------------------------------------
    # 5. Run Training Loop (Integration Test)
    # --------------------------------------------------------------------------
    print("[5] Running Training Loop (Integration Test)...")
    print(
        "    This effectively tests the entire training stack including loss, optimizer, and validation."
    )

    # train_model() uses Config parameters we patched earlier.
    # It will use the real metadata files but limit processing via Config.DEBUG_DATA_LIMIT.
    train_model()

    # Verify artifact creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), "Training failed to produce 'best_model.pth'"

    print(f"    Training complete. Model saved to {best_model_path}")
    print("    Training integration test passed.\n")

    # --------------------------------------------------------------------------
    # 6. Run Inference Pipeline (Integration Test)
    # --------------------------------------------------------------------------
    print("[6] Running Inference Pipeline...")

    predictor = Predictor()
    predictor.run()

    # Verify submission file
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), "Inference failed to produce submission file"

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission generated with {len(df_sub)} rows.")

    # Check format
    assert "BraTS21ID" in df_sub.columns, "Submission missing BraTS21ID column"
    assert "MGMT_value" in df_sub.columns, "Submission missing MGMT_value column"
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print("    Inference integration test passed.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
