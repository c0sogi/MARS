import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.features import generate_feature_matrix
from library.dataset import get_dataset, SeismicDataset
from library.model_lgbm import run_lgbm_cv
from library.model_resnet import run_resnet_cv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata(n_train=20, n_val=10, n_test=10):
    """
    Creates small subsets of the original metadata files to allow
    the demonstration to run quickly.
    """
    print("Creating mini metadata files for demonstration...")

    # Define paths
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Read original metadata
    # We assume the metadata exists as per the problem description
    df_train = pd.read_csv(os.path.join("./metadata", "train.csv"))
    df_val = pd.read_csv(os.path.join("./metadata", "val.csv"))
    df_test = pd.read_csv(os.path.join("./metadata", "test.csv"))

    # Sample subsets
    df_train_mini = df_train.head(n_train).copy()
    df_val_mini = df_val.head(n_val).copy()
    df_test_mini = df_test.head(n_test).copy()

    # Save to working directory
    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    print(f"Mini Train: {len(df_train_mini)} rows")
    print(f"Mini Val: {len(df_val_mini)} rows")
    print(f"Mini Test: {len(df_test_mini)} rows")

    return mini_train_path, mini_val_path, mini_test_path


def configure_demo_settings(mini_train_path, mini_val_path, mini_test_path):
    """
    Monkey-patches the Config class to use mini datasets and
    fast training hyperparameters.
    """
    print("\nConfiguring settings for fast demonstration...")

    # 1. Override Paths
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    # Use a separate cache dir for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_cache"
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )

    # 2. Override Global Settings
    Config.N_FOLDS = 2  # Reduce folds
    Config.NUM_WORKERS = 2

    # 3. Override LightGBM Params
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["num_leaves"] = 8

    # 4. Override ResNet Params
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.RESNET_BASE_FILTERS = 16  # Smaller model

    # Ensure working dir exists
    Config.setup()


def demo_feature_engineering():
    """
    Demonstrates feature extraction using library.features.
    """
    print("\n=== Demo: Feature Engineering ===")

    # Generate features for the mini training set
    # load_cached_data=False forces re-computation
    df_features = generate_feature_matrix(
        Config.TRAIN_METADATA, load_cached_data=False, split_name="train"
    )

    print("Feature Matrix Generated.")
    print(f"Shape: {df_features.shape}")

    # Validation
    # We expect columns for segment_id, time_to_eruption, plus features per sensor
    # 10 sensors * (approx 15 features each) -> > 150 columns
    assert "segment_id" in df_features.columns
    assert "time_to_eruption" in df_features.columns
    assert df_features.shape[0] == 20  # Matches n_train in create_mini_metadata
    assert df_features.shape[1] > 100

    print("Feature Engineering logic verified.")


def demo_lgbm_pipeline():
    """
    Demonstrates the LightGBM training pipeline.
    """
    print("\n=== Demo: LightGBM Pipeline ===")

    # Run the full CV pipeline
    # This uses the Config paths we patched earlier
    submission_df = run_lgbm_cv(load_cached_data=False)

    print("LGBM Pipeline Completed.")
    print(submission_df.head())

    # Validation
    assert isinstance(submission_df, pd.DataFrame)
    assert "segment_id" in submission_df.columns
    assert "time_to_eruption" in submission_df.columns
    assert len(submission_df) == 10  # Matches n_test in create_mini_metadata

    print("LGBM Pipeline logic verified.")


def demo_dataset_loading():
    """
    Demonstrates the SeismicDataset and raw data loading.
    """
    print("\n=== Demo: Dataset & Preprocessing ===")

    # Load raw dataset (train split)
    ds = get_dataset(Config.TRAIN_METADATA, "train", load_cached_data=False)

    print(f"Dataset loaded with {len(ds)} samples.")

    # Check raw data shape
    # (N, SEQ_LEN, NUM_SENSORS)
    assert ds.data.ndim == 3
    assert ds.data.shape[1] == Config.SEQ_LEN
    assert ds.data.shape[2] == Config.NUM_SENSORS

    # Check __getitem__ output (preprocessing happens here)
    # Should return (Tensor(10, 60001), Tensor(target))
    x, y = ds[0]

    print(f"Sample Input Shape (Channels, Time): {x.shape}")
    print(f"Sample Target: {y}")

    # Validation
    assert isinstance(x, torch.Tensor)
    assert x.shape == (Config.NUM_SENSORS, Config.SEQ_LEN)  # (10, 60001)
    assert not torch.isnan(x).any(), "NaNs found in tensor"

    print("Dataset logic verified.")


def demo_resnet_pipeline():
    """
    Demonstrates the ResNet training pipeline.
    """
    print("\n=== Demo: ResNet Pipeline ===")

    # Run the full CV pipeline
    # This uses the Config paths we patched earlier
    submission_df = run_resnet_cv(load_cached_data=False)

    print("ResNet Pipeline Completed.")
    print(submission_df.head())

    # Validation
    assert isinstance(submission_df, pd.DataFrame)
    assert "segment_id" in submission_df.columns
    assert "time_to_eruption" in submission_df.columns
    assert len(submission_df) == 10  # Matches n_test

    print("ResNet Pipeline logic verified.")


if __name__ == "__main__":
    # 1. Reproducibility
    seed_everything(42)

    # 2. Setup Environment
    # Ensure we are using the ./working directory for temporary files
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Create mini metadata to speed up the demo
    mini_train, mini_val, mini_test = create_mini_metadata(
        n_train=20, n_val=10, n_test=10
    )

    # Apply configuration overrides
    configure_demo_settings(mini_train, mini_val, mini_test)

    # 3. Run Demonstrations
    try:
        demo_feature_engineering()
        demo_lgbm_pipeline()
        demo_dataset_loading()
        demo_resnet_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nERROR during demonstration: {e}")
        raise e
