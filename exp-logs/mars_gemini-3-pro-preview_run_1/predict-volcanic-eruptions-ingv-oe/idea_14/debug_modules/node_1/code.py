import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureEngineer
from library.dataset import SeismicDataset
from library.models import ScalarFusedEfficientNet, LightGBMWrapper
from library.train_eval import run_training


def setup_demo_config():
    """
    Overrides Config parameters to run a fast, minimal demonstration.
    Redirects outputs to a new 'demo_run' directory.
    """
    print("Setting up demo configuration...")

    # 1. Enable Debug Mode for small data subsets
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples

    # 2. Reduce Training Intensity
    Config.N_FOLDS = 2  # Only 2 folds
    Config.EPOCHS = 1  # Only 1 epoch for CNN
    Config.BATCH_SIZE = 4  # Small batch size
    Config.LGB_PARAMS["n_estimators"] = 10  # Few trees for LGBM
    Config.LGB_EARLY_STOPPING_ROUNDS = 5

    # 3. Redirect Paths to a clean demo directory
    # This ensures we generate fresh features and don't overwrite main work
    demo_dir = os.path.join("./working", "demo_run")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.TABULAR_TRAIN_CACHE = os.path.join(demo_dir, "train_features.parquet")
    Config.TABULAR_VAL_CACHE = os.path.join(demo_dir, "val_features.parquet")
    Config.TABULAR_TEST_CACHE = os.path.join(demo_dir, "test_features.parquet")

    Config.SPECTROGRAM_TRAIN_DIR = os.path.join(demo_dir, "spectrograms_train")
    Config.SPECTROGRAM_VAL_DIR = os.path.join(demo_dir, "spectrograms_val")
    Config.SPECTROGRAM_TEST_DIR = os.path.join(demo_dir, "spectrograms_test")

    Config.GLOBAL_MAX_PATH = os.path.join(demo_dir, "global_max_spectrogram.npy")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    print(f"Working directory set to: {Config.WORKING_DIR}")


def verify_feature_engineering():
    """
    Demonstrates and verifies FeatureEngineer usage.
    """
    print("\n--- Verifying Feature Engineering ---")
    fe = FeatureEngineer()

    # Generate Vision Features (Spectrograms + Scalars)
    # This will create .npy files in Config.SPECTROGRAM_TRAIN_DIR
    fe.process_vision("train", load_cached_data=False)

    # Check if files were created
    files = os.listdir(Config.SPECTROGRAM_TRAIN_DIR)
    print(f"Generated {len(files)} spectrogram files.")
    assert (
        len(files) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} files, found {len(files)}"

    # Generate Tabular Features
    # This will create a parquet file
    df_tab = fe.process_tabular("train", load_cached_data=False)
    print(f"Generated tabular features with shape: {df_tab.shape}")
    assert (
        len(df_tab) == Config.DEBUG_SAMPLE_SIZE
    ), "Tabular features row count mismatch."
    assert os.path.exists(Config.TABULAR_TRAIN_CACHE), "Tabular cache file not found."


def verify_dataset_and_model():
    """
    Demonstrates and verifies SeismicDataset and ScalarFusedEfficientNet usage.
    """
    print("\n--- Verifying Dataset and Model ---")

    # Load metadata (truncated for debug)
    df_meta = pd.read_csv(Config.TRAIN_METADATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Dataset
    dataset = SeismicDataset(
        metadata=df_meta, data_dir=Config.SPECTROGRAM_TRAIN_DIR, mode="train"
    )

    # Instantiate DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Fetch one batch
    images, scalars, targets = next(iter(loader))

    print(
        f"Batch Shapes -> Images: {images.shape}, Scalars: {scalars.shape}, Targets: {targets.shape}"
    )

    # Assertions for shapes
    # Images: (Batch, 20 channels, 128, 128)
    assert images.shape == (4, 20, 128, 128), "Incorrect image tensor shape."
    # Scalars: (Batch, 30 features)
    assert scalars.shape == (4, 30), "Incorrect scalar tensor shape."
    # Targets: (Batch)
    assert targets.shape == (4,), "Incorrect target tensor shape."

    # Instantiate CNN Model
    model = ScalarFusedEfficientNet()
    model.eval()

    # Forward Pass
    with torch.no_grad():
        output = model(images, scalars)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), "Model output shape should be (Batch, 1)"
    print("Dataset and Model verification successful.")


def run_full_pipeline_demo():
    """
    Runs the full training pipeline using the train_eval module.
    This covers LGBM training, CNN training, Stacking, and Submission.
    """
    print("\n--- Running Full Training Pipeline (Demo) ---")

    # We call the main driver function from train_eval.py
    # Since we set Config.DEBUG=True, this will run on the small subset
    run_training()

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"\nSubmission generated successfully at {Config.SUBMISSION_PATH}")
        print(df_sub.head())

        # Verify submission format
        assert (
            "segment_id" in df_sub.columns and "time_to_eruption" in df_sub.columns
        ), "Submission missing required columns."
        assert len(df_sub) > 0, "Submission file is empty."
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Configure for fast demo
    setup_demo_config()

    # 2. Verify Feature Engineering (Unit Test)
    verify_feature_engineering()

    # 3. Verify Dataset and Model (Unit Test)
    verify_dataset_and_model()

    # 4. Run Full Pipeline (Integration Test)
    run_full_pipeline_demo()

    print("\nAll demonstrations completed successfully.")
