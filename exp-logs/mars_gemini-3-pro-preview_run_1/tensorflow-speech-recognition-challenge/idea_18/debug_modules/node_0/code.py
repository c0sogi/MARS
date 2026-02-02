import os
import shutil
import pandas as pd
import torch
import numpy as np
from library.config import Config
from library.dataset import SpeechDataset, get_dataloaders
from library.model import DilatedEfficientNet
from library.trainer import Trainer
from library.utils import set_seed


def create_small_metadata(n_train=100, n_val=50, n_test=50):
    """
    Creates small subsets of the original metadata for rapid demonstration.
    """
    print(f"Creating small metadata subsets in {Config.WORK_DIR}...")

    # Load original metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    # Sample subsets
    # Ensure we include some target labels in the small set for proper class mapping
    df_train_small = df_train.sample(n=min(n_train, len(df_train)), random_state=42)
    df_val_small = df_val.sample(n=min(n_val, len(df_val)), random_state=42)
    df_test_small = df_test.sample(n=min(n_test, len(df_test)), random_state=42)

    # Define paths
    train_path = os.path.join(Config.WORK_DIR, "train_small.csv")
    val_path = os.path.join(Config.WORK_DIR, "val_small.csv")
    test_path = os.path.join(Config.WORK_DIR, "test_small.csv")

    # Save
    df_train_small.to_csv(train_path, index=False)
    df_val_small.to_csv(val_path, index=False)
    df_test_small.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def verify_dataset_logic(train_csv_path):
    """
    Verifies SpeechDataset functionality: loading, transforms, and shapes.
    """
    print("\n=== Verifying Dataset Logic ===")

    # Create a dummy dataframe and class mapping
    df = pd.read_csv(train_csv_path)
    # Mock fine_label column as it is usually added by process_metadata
    df["fine_label"] = df["label"]

    # Create a simple class map
    unique_labels = df["label"].unique()
    class_to_idx = {label: i for i, label in enumerate(unique_labels)}

    # Instantiate Dataset
    dataset = SpeechDataset(df, mode="train", class_to_idx=class_to_idx)

    # Check length
    assert len(dataset) == len(df), "Dataset length mismatch"

    # Check item retrieval
    spec, label_idx = dataset[0]

    # Verify Spectrogram Shape: (Channels, Freq, Time)
    # Config: N_MELS=128. Time depends on DURATION (1.0s) * SR (16000) / HOP (160) approx 101 frames
    expected_freq = Config.N_MELS
    # 16000 / 160 = 100 frames + 1 = 101 usually
    expected_time = int(Config.SR * Config.DURATION / Config.HOP_LENGTH) + 1

    print(f"Spectrogram shape: {spec.shape}")

    assert spec.dim() == 3, "Spectrogram must be 3D (C, F, T)"
    assert spec.size(0) == 1, "Input channel should be 1"
    assert spec.size(1) == expected_freq, f"Expected {expected_freq} mel bands"
    # Allow small variance in time dimension due to padding/rounding
    assert (
        abs(spec.size(2) - expected_time) <= 2
    ), f"Time dimension {spec.size(2)} unexpected"

    assert isinstance(label_idx, (int, np.integer)), "Label index must be integer"
    print("Dataset verification passed.")


def verify_model_logic():
    """
    Verifies Model architecture and forward pass.
    """
    print("\n=== Verifying Model Logic ===")

    num_classes = 31
    model = DilatedEfficientNet(num_classes=num_classes)
    model.eval()

    # Create dummy input: (Batch, 1, Freq, Time)
    batch_size = 4
    freq = Config.N_MELS
    time_steps = 101
    dummy_input = torch.randn(batch_size, 1, freq, time_steps)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output shape: {output.shape}")

    assert output.shape == (batch_size, num_classes), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model output contains NaNs"
    print("Model verification passed.")


def run_demo_training():
    """
    Runs the Trainer with modified config to demonstrate the full pipeline.
    """
    print("\n=== Running Full Training Pipeline (Demo) ===")

    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    # pass load_cached_data=False to force reprocessing of our new small CSVs
    trainer.train(load_cached_data=False)

    # Verify Outputs
    print("\n=== Verifying Artifacts ===")

    expected_files = ["best_model.pth", "swa_model.pth", "submission.csv"]

    for fname in expected_files:
        fpath = os.path.join(Config.WORK_DIR, fname)
        assert os.path.exists(fpath), f"Missing artifact: {fname}"
        print(f"Found {fname}")

    # Verify Submission Format
    sub_df = pd.read_csv(os.path.join(Config.WORK_DIR, "submission.csv"))
    assert list(sub_df.columns) == ["fname", "label"], "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check if labels are valid competition labels
    valid_labels = set(
        Config.TARGET_LABELS + [Config.SILENCE_LABEL, Config.UNKNOWN_LABEL]
    )
    pred_labels = set(sub_df["label"].unique())
    invalid_preds = pred_labels - valid_labels
    assert (
        len(invalid_preds) == 0
    ), f"Found invalid labels in submission: {invalid_preds}"

    print("Pipeline verification passed.")


if __name__ == "__main__":
    # 1. Setup Environment
    set_seed(42)

    # 2. Patch Configuration for Speed
    # We modify the Config class attributes directly before they are used.
    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORK_DIR
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Reduce training intensity
    Config.EPOCHS = 2
    Config.SWA_START_EPOCH = 2  # Start SWA immediately at epoch 2
    Config.BATCH_SIZE = 16  # Small batch for demo
    Config.TARGET_SAMPLES_PER_CLASS = 20  # Minimal balancing

    # 3. Create Small Dataset
    train_csv, val_csv, test_csv = create_small_metadata()
    Config.TRAIN_CSV = train_csv
    Config.VAL_CSV = val_csv
    Config.TEST_CSV = test_csv

    # 4. Run Verifications
    try:
        verify_dataset_logic(train_csv)
        verify_model_logic()
        run_demo_training()
        print("\nSUCCESS: All demonstrations and verifications completed.")
    except Exception as e:
        print(f"\nFAILURE: An error occurred during execution: {e}")
        raise e
