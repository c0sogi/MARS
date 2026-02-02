import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import library components
from library.config import (
    METADATA_DIR,
    SEED,
    NUM_SAMPLES,
    N_MELS,
    INPUT_ROOT,
    IDX_TO_LABEL,
)
from library.utils import set_seed, get_device, EarlyStopping
from library.preprocessing import process_file, get_cache_filename
from library.dataset import HybridAudioDataset
from library.model import HybridDualStreamCRNN
from library.trainer import Trainer
from library.transforms import SpecAugment, RawAudioAugment

# Constants for Demo Execution
DEMO_WORK_DIR = "./working/demo_execution"
DEMO_CACHE_DIR = os.path.join(DEMO_WORK_DIR, "cache")
DEMO_TRAIN_CSV = os.path.join(DEMO_WORK_DIR, "train.csv")
DEMO_VAL_CSV = os.path.join(DEMO_WORK_DIR, "val.csv")
DEMO_TEST_CSV = os.path.join(DEMO_WORK_DIR, "test.csv")
DEMO_MODEL_PATH = os.path.join(DEMO_WORK_DIR, "best_model.pth")
DEMO_SUBMISSION_PATH = os.path.join(DEMO_WORK_DIR, "submission.csv")


def setup_demo_environment():
    """Sets up directories and creates small subset metadata files for speed."""
    print(">>> Setting up demo environment...")

    # Clean up previous runs if any
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Create small subsets (ensure enough samples for a batch)
    subset_train = train_df.head(32)
    subset_val = val_df.head(16)
    subset_test = test_df.head(16)

    # Save subsets
    subset_train.to_csv(DEMO_TRAIN_CSV, index=False)
    subset_val.to_csv(DEMO_VAL_CSV, index=False)
    subset_test.to_csv(DEMO_TEST_CSV, index=False)

    print(
        f"Created subset metadata: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )
    return subset_train, subset_val, subset_test


def demo_preprocessing(dfs):
    """Demonstrates feature extraction and caching mechanism."""
    print("\n>>> Demonstrating Preprocessing (Caching)...")

    # Combine all unique filepaths from the subsets
    all_files = pd.concat(dfs)["filepath"].unique()

    print(f"Processing {len(all_files)} files for the demo...")
    for filepath in all_files:
        # process_file handles loading audio, computing spec, and saving to cache
        # We point it to our demo cache directory
        process_file(filepath, DEMO_CACHE_DIR, load_cached_data=False)

        # Verify cache file exists
        cache_name = get_cache_filename(filepath)
        cache_path = os.path.join(DEMO_CACHE_DIR, cache_name)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Failed to cache file: {cache_path}")

    print("Preprocessing verification successful.")


def demo_dataset_loading():
    """Demonstrates Dataset class usage, transforms, and verifies data shapes."""
    print("\n>>> Demonstrating Dataset and Transforms...")

    # Initialize Augmentations (p=1.0 to ensure they are applied for verification)
    spec_aug = SpecAugment(p=1.0)
    raw_aug = RawAudioAugment(p=1.0)

    # Initialize Dataset with demo metadata and cache
    ds = HybridAudioDataset(
        metadata_file=DEMO_TRAIN_CSV,
        cache_dir=DEMO_CACHE_DIR,
        spec_augment=spec_aug,
        raw_augment=raw_aug,
        is_test=False,
    )

    print(f"Dataset length: {len(ds)}")

    # Fetch a sample
    spec, wave, label = ds[0]

    # Verify Shapes
    # Spec: (3, N_MELS, Time)
    # Time is approx 101 for 1 sec audio with hop 160 (16000/160 + padding)
    print(f"Sample Spec Shape: {spec.shape}")
    print(f"Sample Wave Shape: {wave.shape}")
    print(f"Sample Label Index: {label}")

    # Assertions
    assert spec.dim() == 3, "Spectrogram must be 3D (Channels, Freq, Time)"
    assert spec.shape[0] == 3, "Spectrogram must have 3 channels (Multi-resolution)"
    assert spec.shape[1] == N_MELS, f"Spectrogram freq dim must be {N_MELS}"
    assert wave.dim() == 2, "Waveform must be 2D (Channels, Time)"
    assert wave.shape[0] == 1, "Waveform must be Mono"
    assert wave.shape[1] == NUM_SAMPLES, f"Waveform length must be {NUM_SAMPLES}"
    assert isinstance(label, int), "Label must be an integer index"

    print("Dataset verification successful.")
    return ds


def demo_model_inference(dataset):
    """Demonstrates Model initialization and forward pass on a batch."""
    print("\n>>> Demonstrating Model Forward Pass...")

    device = get_device()
    model = HybridDualStreamCRNN().to(device)

    # Create a DataLoader to get a batch
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    spec_batch, wave_batch, label_batch = next(iter(loader))

    spec_batch = spec_batch.to(device)
    wave_batch = wave_batch.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(spec_batch, wave_batch)

    print(f"Output Batch Shape: {outputs.shape}")

    # Assertions
    # Shape should be (Batch_Size, Num_Classes) -> (4, 12)
    assert outputs.shape == (
        4,
        12,
    ), f"Expected output shape (4, 12), got {outputs.shape}"

    print("Model forward pass verification successful.")
    return model


def demo_training_loop(model):
    """Demonstrates the training loop using the Trainer class."""
    print("\n>>> Demonstrating Training Loop...")

    device = get_device()

    # Setup Datasets and Loaders for Demo
    train_ds = HybridAudioDataset(DEMO_TRAIN_CSV, cache_dir=DEMO_CACHE_DIR)
    val_ds = HybridAudioDataset(DEMO_VAL_CSV, cache_dir=DEMO_CACHE_DIR)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Setup Training Components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Early Stopping (configured to save to demo dir)
    early_stopping = EarlyStopping(patience=2, verbose=True, path=DEMO_MODEL_PATH)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        early_stopping=early_stopping,
    )

    # Run Fit (2 Epochs for speed)
    print("Starting training (2 epochs)...")
    trainer.fit(train_loader, val_loader, num_epochs=2)

    # Check if model checkpoint was created
    if os.path.exists(DEMO_MODEL_PATH):
        print(f"Checkpoint successfully saved at {DEMO_MODEL_PATH}")
    else:
        raise AssertionError("Model checkpoint was not created during training!")

    print("Training loop verification successful.")


def demo_submission_generation():
    """Demonstrates generating predictions using the trained model on test data."""
    print("\n>>> Demonstrating Submission Generation...")

    device = get_device()

    # Load Test Dataset (is_test=True)
    test_ds = HybridAudioDataset(DEMO_TEST_CSV, cache_dir=DEMO_CACHE_DIR, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Load Model from Checkpoint
    model = HybridDualStreamCRNN().to(device)
    if not os.path.exists(DEMO_MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {DEMO_MODEL_PATH}")

    checkpoint = torch.load(DEMO_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []

    # Inference Loop
    with torch.no_grad():
        for spec, wave, _ in test_loader:
            spec = spec.to(device)
            wave = wave.to(device)

            outputs = model(spec, wave)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Verify prediction count matches dataset
    assert len(predictions) == len(
        test_ds
    ), "Number of predictions does not match test set size"

    # Map indices to labels and create DataFrame
    pred_labels = [IDX_TO_LABEL[p] for p in predictions]

    sub_df = pd.DataFrame(
        {"fname": test_ds.df["filepath"].apply(os.path.basename), "label": pred_labels}
    )

    # Save Submission
    sub_df.to_csv(DEMO_SUBMISSION_PATH, index=False)

    print(f"Submission generated with {len(sub_df)} rows.")
    print(sub_df.head())
    print("Submission generation verification successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(SEED)

    # 1. Setup Environment and Data Subsets
    dfs = setup_demo_environment()

    # 2. Preprocess Data (Feature Extraction & Caching)
    demo_preprocessing(dfs)

    # 3. Initialize Dataset and Verify Transforms
    ds = demo_dataset_loading()

    # 4. Initialize Model and Verify Forward Pass
    model = demo_model_inference(ds)

    # 5. Run Training Loop
    demo_training_loop(model)

    # 6. Generate Submission
    demo_submission_generation()

    print("\n>>> All demonstrations completed successfully.")
