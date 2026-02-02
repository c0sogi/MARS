import os
import sys
import pandas as pd
import numpy as np
import warnings
import json
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data_manager import DataBuilder
from library.model_factory import DualStreamModel
from library.evaluation import Evaluator


def create_mini_metadata():
    """
    Creates small subsets of the metadata files to allow the pipeline
    to run quickly for demonstration purposes.
    """
    print("Creating mini-datasets for rapid demonstration...")

    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Select a small number of plays (groups)
    train_plays = df_train["game_play"].unique()[:2]
    val_plays = df_val["game_play"].unique()[:1]

    mini_train = df_train[df_train["game_play"].isin(train_plays)].copy()
    mini_val = df_val[df_val["game_play"].isin(val_plays)].copy()

    # Define paths for mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")

    # Save to disk
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)

    return mini_train_path, mini_val_path


def configure_demo_settings(mini_train_path, mini_val_path):
    """
    Overrides Config settings to use mini datasets and faster model parameters.
    """
    print("Overriding configuration for demo...")

    # Override Metadata Paths
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path

    # We keep Test Metadata as is, or we could slice it too.
    # For this demo, let's just slice the test metadata in memory if needed,
    # but the FeatureGenerator reads from disk. Let's create a mini test too.
    df_test = pd.read_csv(Config.TEST_META_PATH)
    test_plays = df_test["game_play"].unique()[:1]
    mini_test = df_test[df_test["game_play"].isin(test_plays)].copy()
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")
    mini_test.to_csv(mini_test_path, index=False)
    Config.TEST_META_PATH = mini_test_path

    # Override Model Hyperparameters for Speed
    # Reduce estimators to a minimal number to verify the training loop works
    Config.XGB_PARAMS_A["n_estimators"] = 10
    Config.XGB_PARAMS_A["max_depth"] = 3
    Config.XGB_PARAMS_B["n_estimators"] = 10
    Config.XGB_PARAMS_B["max_depth"] = 3

    # Force CPU to avoid any potential GPU init overhead for such small data,
    # though the library defaults to gpu_hist.
    # If GPU is available (A100), gpu_hist is fine, but 'hist' is safer for tiny datasets.
    Config.XGB_PARAMS_A["tree_method"] = "hist"
    Config.XGB_PARAMS_B["tree_method"] = "hist"
    Config.XGB_PARAMS_A["device"] = "cpu"
    Config.XGB_PARAMS_B["device"] = "cpu"

    # Disable caching to ensure we run the generation logic
    # We do this by ensuring the cache files don't exist or by passing load_cached_data=False
    # We will pass load_cached_data=False in the calls.


def validate_data_structure(data_dict, split_name):
    """
    Validates the structure of the data dictionary returned by DataBuilder.
    """
    print(f"Validating {split_name} data structure...")
    assert "stream_a" in data_dict, f"{split_name} missing stream_a"
    assert "stream_b" in data_dict, f"{split_name} missing stream_b"

    for stream in ["stream_a", "stream_b"]:
        components = data_dict[stream]
        assert "X" in components, f"{stream} missing X"
        assert "y" in components, f"{stream} missing y"
        assert "ids" in components, f"{stream} missing ids"

        n_samples = len(components["y"])
        assert len(components["X"]) == n_samples, f"X and y length mismatch in {stream}"
        assert (
            len(components["ids"]) == n_samples
        ), f"ids and y length mismatch in {stream}"

        if n_samples > 0:
            print(
                f"  {stream}: {n_samples} samples. Features: {components['X'].shape[1]}"
            )


def run_pipeline():
    # 1. Setup
    set_seed(42)
    mini_train_path, mini_val_path = create_mini_metadata()
    configure_demo_settings(mini_train_path, mini_val_path)

    # 2. Data Building
    print("\n=== Data Preparation Phase ===")
    data_builder = DataBuilder()

    # Get Train Data (Force regeneration to demonstrate FeatureGenerator)
    print("Generating Training Data...")
    train_data = data_builder.get_stream_data(split="train", load_cached_data=False)
    validate_data_structure(train_data, "Train")

    # Get Validation Data
    print("Generating Validation Data...")
    val_data = data_builder.get_stream_data(split="validation", load_cached_data=False)
    validate_data_structure(val_data, "Validation")

    # 3. Model Training
    print("\n=== Model Training Phase ===")
    model = DualStreamModel()
    model.fit(train_data, val_data)

    # Verify artifacts
    assert model.model_a is not None, "Model A failed to train"
    assert model.model_b is not None, "Model B failed to train"
    assert 0.0 < model.threshold_a < 1.0, "Threshold A not optimized"
    assert 0.0 < model.threshold_b < 1.0, "Threshold B not optimized"

    # Check if files were saved
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "model_a.json")
    ), "Model A file not saved"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "model_meta.json")
    ), "Model metadata not saved"

    # 4. Inference
    print("\n=== Inference Phase ===")
    # Get Test Data
    print("Generating Test Data...")
    test_data = data_builder.get_stream_data(split="test", load_cached_data=False)
    validate_data_structure(test_data, "Test")

    # Predict
    submission = model.predict(test_data)

    # 5. Submission Validation
    print("\n=== Validating Submission ===")
    print(submission.head())

    # Check columns
    assert "contact_id" in submission.columns, "Submission missing contact_id"
    assert "contact" in submission.columns, "Submission missing contact"

    # Check values
    assert submission["contact"].isin([0, 1]).all(), "Predictions must be binary"

    # Check length matches test input (stream A + stream B unique IDs)
    n_test_a = len(test_data["stream_a"]["ids"])
    n_test_b = len(test_data["stream_b"]["ids"])
    # Note: streams are disjoint by definition (player-player vs player-ground),
    # but let's verify total count roughly matches.
    # The submission drops duplicates, but inputs shouldn't have duplicates across streams.
    print(f"Input samples: {n_test_a + n_test_b}")
    print(f"Submission rows: {len(submission)}")

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\nPipeline demonstration completed successfully.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
