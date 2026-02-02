import os
import glob
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import provided library modules
from library.config import Config
from library.dataset import SpeechCommandDataset, get_dataloaders, LABELS
from library.model import EfficientNetV2Audio
from library.trainer import Trainer
from library.utils import set_seed, load_checkpoint, get_device


def setup_demo_environment():
    """
    Prepares the environment for a fast demo run:
    1. Cleans up existing cache.
    2. Creates subset metadata files.
    3. Updates Config to use these subsets and fast training parameters.
    """
    print("--- Setting up Demo Environment ---")

    # 1. Clean up existing cache in working directory to force reload
    # The dataset class caches .npy files. We want to generate new ones from our subsets.
    cache_files = glob.glob(os.path.join(Config.WORKING_DIR, "*.npy"))
    for f in cache_files:
        try:
            os.remove(f)
        except OSError:
            pass
    print(f"Cleaned {len(cache_files)} cache files.")

    # 2. Create Subsets of Metadata
    # We create temporary CSVs in the working directory
    subset_dir = os.path.join(Config.WORKING_DIR, "demo_subsets")
    os.makedirs(subset_dir, exist_ok=True)

    # Helper to sample and save
    def create_subset(src_path, dst_name, n=50):
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found. Skipping.")
            return None
        df = pd.read_csv(src_path)
        # Sample n rows, or all if less than n
        n = min(n, len(df))
        df_subset = df.sample(n=n, random_state=Config.SEED).reset_index(drop=True)
        dst_path = os.path.join(subset_dir, dst_name)
        df_subset.to_csv(dst_path, index=False)
        return dst_path

    train_subset_path = create_subset(Config.TRAIN_CSV, "train.csv", n=64)
    val_subset_path = create_subset(Config.VAL_CSV, "val.csv", n=32)
    test_subset_path = create_subset(Config.TEST_CSV, "test.csv", n=32)

    print("Created metadata subsets.")

    # 3. Update Config
    # We modify the static Config class directly
    Config.TRAIN_CSV = train_subset_path
    Config.VAL_CSV = val_subset_path
    Config.TEST_CSV = test_subset_path

    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.PATIENCE = 2

    # Update submission path for demo
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    print("Config updated for fast execution.")


def verify_dataset_logic():
    """
    Verifies that the Dataset class loads data correctly and produces expected shapes.
    """
    print("\n--- Verifying Dataset Logic ---")

    # Instantiate dataset with load_cached_data=False to force processing of our new subsets
    # (Though we deleted cache, explicit False is safer for demo logic if called directly)
    ds = SpeechCommandDataset("train", load_cached_data=False)

    print(f"Dataset length: {len(ds)}")
    assert len(ds) > 0, "Dataset should not be empty."

    # Fetch one sample
    spec, label = ds[0]

    # Verify Spectrogram Shape: (1, n_mels, time)
    # n_mels is 128 (Config), time depends on hop length.
    # 16000 samples / 160 hop = 100 frames + padding/centering usually ~101
    print(f"Sample shape: {spec.shape}, Label: {label}")

    assert spec.dim() == 3, "Spectrogram should be 3D (C, F, T)"
    assert spec.size(0) == 1, "Channel dimension should be 1"
    assert spec.size(1) == Config.N_MELS, f"Freq dimension should be {Config.N_MELS}"
    assert isinstance(
        label, (int, np.integer, torch.Tensor)
    ), "Label should be an integer"

    print("Dataset verification passed.")


def verify_model_logic():
    """
    Verifies that the Model instantiates and performs a forward pass.
    """
    print("\n--- Verifying Model Logic ---")

    device = get_device()
    model = EfficientNetV2Audio(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)
    model.eval()

    # Create dummy input: (Batch, 1, Freq, Time)
    # Time dimension approx 101 for 1 sec audio with current STFT settings
    dummy_input = torch.randn(2, 1, Config.N_MELS, 101).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")

    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"

    print("Model verification passed.")


def run_training_demo():
    """
    Runs the Trainer to demonstrate the training loop.
    """
    print("\n--- Running Training Demo ---")

    # Initialize Trainer
    # This will load data using the updated Config paths
    trainer = Trainer()

    # Run Fit
    trainer.fit()

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("Training demo completed successfully.")


def run_inference_demo():
    """
    Runs inference using the trained model and generates a submission file.
    """
    print("\n--- Running Inference Demo ---")

    device = get_device()

    # 1. Load Model
    model = EfficientNetV2Audio(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)

    checkpoint = load_checkpoint(model, filepath=Config.BEST_MODEL_PATH, device=device)
    assert checkpoint is not None, "Failed to load checkpoint."

    model.eval()

    # 2. Load Test Data
    # We use the test loader created via get_dataloaders
    # Note: get_dataloaders returns (train, val, test)
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    predictions = []
    fnames = []

    # 3. Inference Loop
    # We need to access filenames. The dataset stores them.
    # The loader returns (images, labels). Labels are dummy in test.
    # We need to access the dataset within the loader to get filenames corresponding to indices,
    # but the standard loader shuffles (if set) or batches.
    # The provided dataset implementation doesn't return fnames in __getitem__.
    # However, the test loader in get_dataloaders has shuffle=False.
    # We can iterate the dataset directly or rely on order preservation.

    print(f"Running inference on {len(test_loader.dataset)} test samples...")

    current_idx = 0
    test_dataset = test_loader.dataset

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            batch_size = images.size(0)

            # Get filenames for this batch
            # Since shuffle=False, we can slice the dataset's dataframe or fnames array
            batch_fnames = test_dataset.df.iloc[current_idx : current_idx + batch_size][
                "fname"
            ].values

            predictions.extend(preds)
            fnames.extend(batch_fnames)

            current_idx += batch_size

    # 4. Generate Submission
    # Map indices back to labels
    # LABELS list is in dataset.py. We imported it.
    pred_labels = [LABELS[p] for p in predictions]

    df_sub = pd.DataFrame({"fname": fnames, "label": pred_labels})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())

    # Verify format
    assert df_sub.shape[1] == 2, "Submission should have 2 columns"
    assert "fname" in df_sub.columns and "label" in df_sub.columns, "Incorrect columns"


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup Environment (Subset Data, Config Override)
    setup_demo_environment()

    # 2. Verify Components
    verify_dataset_logic()
    verify_model_logic()

    # 3. Run Training
    run_training_demo()

    # 4. Run Inference
    run_inference_demo()

    print("\n=== Full Demo Completed Successfully ===")
