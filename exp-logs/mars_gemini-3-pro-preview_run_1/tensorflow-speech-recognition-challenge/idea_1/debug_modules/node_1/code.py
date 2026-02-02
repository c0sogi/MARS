import sys
import types
import importlib.util

# 1. Mock tqdm to suppress progress bars (must be done before importing library modules)
tqdm_module = types.ModuleType("tqdm")


def tqdm_func(iterable, *args, **kwargs):
    return iterable


tqdm_module.tqdm = tqdm_func
# Fix: Set __spec__ to satisfy importlib/torch.dynamo introspection
tqdm_module.__spec__ = importlib.util.spec_from_loader("tqdm", loader=None)
sys.modules["tqdm"] = tqdm_module

import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.dataset import SpeechCommandsDataset
from library.model import SpectroCNN
from library.trainer import run_training


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_demo_data():
    """
    Creates small subsets of the metadata to ensure the demo runs quickly.
    Returns paths to the new subset CSV files.
    """
    print("Preparing small dataset subsets for demonstration...")

    # Load original metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    # Create Train Subset: ensure we have samples for all targets, unknown, and silence
    train_subset = []

    # 1. Targets (5 samples each)
    targets = [l for l in Config.LABELS if l not in ["silence", "unknown"]]
    for label in targets:
        samples = df_train[df_train["label"] == label].head(5)
        train_subset.append(samples)

    # 2. Unknown (5 samples)
    unknowns = df_train[df_train["label"] == "unknown"].head(5)
    train_subset.append(unknowns)

    # 3. Silence (5 samples)
    silences = df_train[df_train["label"] == "silence"].head(5)
    train_subset.append(silences)

    df_train_small = pd.concat(train_subset, ignore_index=True)

    # Create Val and Test Subsets (Random small samples)
    df_val_small = df_val.sample(n=20, random_state=42)
    df_test_small = df_test.sample(n=20, random_state=42)

    # Save to working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    train_path = os.path.join(Config.WORKING_DIR, "train_small.csv")
    val_path = os.path.join(Config.WORKING_DIR, "val_small.csv")
    test_path = os.path.join(Config.WORKING_DIR, "test_small.csv")

    df_train_small.to_csv(train_path, index=False)
    df_val_small.to_csv(val_path, index=False)
    df_test_small.to_csv(test_path, index=False)

    print(
        f"Created subsets: Train={len(df_train_small)}, Val={len(df_val_small)}, Test={len(df_test_small)}"
    )

    return train_path, val_path, test_path


def configure_for_demo(train_path, val_path, test_path):
    """
    Updates Config to use the small datasets and fast training parameters.
    """
    # Update file paths
    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path

    # Use a separate cache dir for demo to avoid conflicts with real training
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")

    # Optimize hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.UNKNOWN_TRAIN_SAMPLE_COUNT = 10  # Downsample 'unknown' class heavily
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in small script


def verify_components():
    """
    Verifies that Dataset and Model components work as expected.
    """
    print("\nVerifying components...")

    # 1. Test Dataset
    # load_cached_data=False forces the dataset to read our new small CSVs
    ds = SpeechCommandsDataset(mode="train", load_cached_data=False)
    print(f"Dataset initialized. Effective Length (after balancing): {len(ds)}")

    # Check item structure
    spec, label = ds[0]

    # Expected shape: [1, N_MELS, Time]
    # Time dimension depends on padding/stft, usually around 101 for 1 sec at 16k sr
    assert spec.ndim == 3, f"Spectrogram should be 3D, got {spec.shape}"
    assert spec.shape[0] == 1, f"Expected 1 channel, got {spec.shape[0]}"
    assert (
        spec.shape[1] == Config.N_MELS
    ), f"Expected {Config.N_MELS} mels, got {spec.shape[1]}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    print(f"Sample shape verified: {spec.shape}")

    # 2. Test Model
    model = SpectroCNN(num_classes=Config.NUM_CLASSES)
    model.eval()

    # Create dummy batch
    batch_size = 4
    time_dim = spec.shape[2]
    dummy_input = torch.randn(batch_size, 1, Config.N_MELS, time_dim)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(batch_size, Config.NUM_CLASSES)}, got {output.shape}"

    print("Model forward pass verified.")


if __name__ == "__main__":
    set_seed(42)

    # 1. Prepare Data Subsets
    train_csv, val_csv, test_csv = prepare_demo_data()

    # 2. Apply Configuration Overrides
    configure_for_demo(train_csv, val_csv, test_csv)

    # 3. Verify Logic
    verify_components()

    # 4. Run Training Pipeline
    # This executes the training loop, validation, and generates submission.csv
    print("\nStarting Training Pipeline Demonstration...")
    run_training(load_cached_data=False)

    # 5. Verify Submission Output
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"\nSubmission generated successfully at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")

        # Verify row count matches our test subset size
        test_len = len(pd.read_csv(test_csv))
        assert (
            len(df_sub) == test_len
        ), f"Submission rows {len(df_sub)} != Test rows {test_len}"

        print("Sample Predictions:")
        print(df_sub.head())
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nDemonstration completed successfully.")
