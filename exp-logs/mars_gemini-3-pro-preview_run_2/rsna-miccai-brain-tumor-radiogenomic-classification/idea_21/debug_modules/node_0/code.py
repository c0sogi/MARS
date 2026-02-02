import os
import sys
import pandas as pd
import numpy as np
import torch

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_io import read_dicom_robust, resize_image
from library.dataset import BraTSDataset
from library.model import AsymmetricEfficientNet
from library.engine import run_training, generate_submission


def create_demo_subsets():
    """
    Creates small subsets of the metadata files in the working directory
    to allow for a quick demonstration run.
    """
    print("Creating data subsets for demonstration...")

    # Load original metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (e.g., 10 samples for train, 4 for val, 4 for test)
    # This ensures the code runs in seconds rather than hours
    demo_train = df_train.head(10).copy()
    demo_val = df_val.head(4).copy()
    demo_test = df_test.head(4).copy()

    # Save to working directory
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path


def verify_data_components(train_csv_path):
    """
    Verifies Data I/O, ROI selection, and Dataset logic.
    """
    print("\n--- Verifying Data Components ---")

    # 1. Verify Low-Level I/O
    df = pd.read_csv(train_csv_path)
    sample_row = df.iloc[0]
    flair_path_rel = sample_row["path_FLAIR"]
    flair_dir = os.path.join(Config.INPUT_DIR, flair_path_rel)

    # Find a real file
    files = sorted(os.listdir(flair_dir))
    if files:
        sample_file = os.path.join(flair_dir, files[0])
        img = read_dicom_robust(sample_file)

        # Assertions
        assert isinstance(img, np.ndarray), "read_dicom_robust returned wrong type"
        assert img.dtype == np.uint16, f"Expected uint16, got {img.dtype}"

        resized = resize_image(img, Config.IMG_SIZE)
        assert resized.shape == (Config.IMG_SIZE, Config.IMG_SIZE), "Resize failed"
        assert resized.dtype == np.float32, "Resize should return float32"
        print("Low-level I/O verified.")

    # 2. Verify Dataset
    # We use the subset dataframe
    ds = BraTSDataset(df, phase="train", load_cached_data=False)

    # Fetch one item
    image_tensor, target = ds[0]

    # Check Shape: (12, 224, 224)
    # 4 modalities * 3 slices = 12 channels
    assert image_tensor.shape == (
        12,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Dataset tensor shape mismatch. Expected (12, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {image_tensor.shape}"

    # Check Normalization (Independent Min-Max -> [0, 1])
    assert (
        image_tensor.min() >= 0.0 and image_tensor.max() <= 1.0
    ), "Normalization failed. Values out of [0, 1] range."

    # Check Target
    assert isinstance(target, torch.Tensor), "Target is not a tensor"

    print("Dataset logic verified.")
    return image_tensor.unsqueeze(0)  # Return batch of size 1 for model test


def verify_model_components(sample_input):
    """
    Verifies Model architecture and forward pass.
    """
    print("\n--- Verifying Model Components ---")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = AsymmetricEfficientNet().to(device)
    model.eval()

    with torch.no_grad():
        output = model(sample_input.to(device))

    # Check Output Shape: (Batch_Size, 1)
    assert output.shape == (
        1,
        1,
    ), f"Model output shape mismatch. Expected (1, 1), got {output.shape}"

    print("Model architecture verified.")


def run_demo_pipeline():
    """
    Runs the full training and inference pipeline on the demo subset.
    """
    print("\n--- Running Demo Pipeline ---")

    # Initialize Logger
    logger = get_logger(os.path.join(Config.WORKING_DIR, "demo_log.txt"))

    # Run Training
    # The engine uses Config paths, which we overrode globally
    best_auc = run_training(logger=logger)

    print(f"Training complete. Best AUC on demo subset: {best_auc}")

    # Verify Model Artifact
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."

    # Run Inference
    df_submission = generate_submission(logger=logger)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not saved."
    assert (
        len(df_submission) == 4
    ), "Submission row count mismatch (expected 4 for demo)."
    assert "BraTS21ID" in df_submission.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in df_submission.columns, "Missing MGMT_value column"

    # Check ID format (should be 5-digit string based on engine.py logic)
    sample_id = df_submission.iloc[0]["BraTS21ID"]
    assert (
        isinstance(sample_id, str) and len(sample_id) == 5
    ), f"Submission ID format incorrect. Expected 5-digit string, got {sample_id}"

    print("Inference pipeline verified.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Override Config for Speed
    # We modify the Config class attributes directly for this runtime session
    demo_train_csv, demo_val_csv, demo_test_csv = create_demo_subsets()

    Config.TRAIN_CSV = demo_train_csv
    Config.VAL_CSV = demo_val_csv
    Config.TEST_CSV = demo_test_csv

    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # No multiprocessing overhead
    Config.ROI_CACHE_FILE = os.path.join(Config.WORKING_DIR, "demo_roi_cache.parquet")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # 3. Verify Components
    sample_batch = verify_data_components(demo_train_csv)
    verify_model_components(sample_batch)

    # 4. Run Pipeline
    run_demo_pipeline()

    print("\nAll demonstrations passed successfully.")
