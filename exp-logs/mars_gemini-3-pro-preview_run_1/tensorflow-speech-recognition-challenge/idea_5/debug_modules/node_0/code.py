import os
import sys
import shutil
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Ensure the current directory is in the path for module imports
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, MetricMonitor
from library.dataset import SpeechCommandDataset, get_dataset
from library.model import get_model, DilatedConvNeXt
from library.train import train_one_epoch, validate


def run_demo():
    print("=== Starting Speech Command Recognition Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Setup & Configuration
    # ------------------------------------------------------------------------
    print("[1] Setting up Configuration...")

    # Define a Demo Config that overrides paths and hyperparams for speed
    class DemoConfig(Config):
        # Use a separate working directory for the demo
        WORKING_DIR = "./working/demo_run"

        # Point to where we will create small metadata files
        TRAIN_METADATA_PATH = os.path.join(WORKING_DIR, "train_small.csv")
        VAL_METADATA_PATH = os.path.join(WORKING_DIR, "val_small.csv")
        TEST_METADATA_PATH = os.path.join(WORKING_DIR, "test_small.csv")

        # Hyperparameters for fast execution
        EPOCHS = 1
        BATCH_SIZE = 4
        NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

        # Model saving
        BEST_MODEL_PATH = os.path.join(WORKING_DIR, "demo_best_model.pth")
        SUBMISSION_PATH = os.path.join(WORKING_DIR, "demo_submission.csv")

    config = DemoConfig()
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.SEED)
    print("    Configuration initialized and seed set.")

    # ------------------------------------------------------------------------
    # 2. Prepare Dummy Data (Subset of Real Data)
    # ------------------------------------------------------------------------
    print("\n[2] Preparing Data Subsets...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample a small fraction for the demo (ensure we have some of each class if possible)
    # We take 50 samples for train, 20 for val, 20 for test
    train_small = orig_train.sample(n=50, random_state=config.SEED).reset_index(
        drop=True
    )
    val_small = orig_val.sample(n=20, random_state=config.SEED).reset_index(drop=True)
    test_small = orig_test.sample(n=20, random_state=config.SEED).reset_index(drop=True)

    # Save to the demo paths defined in DemoConfig
    train_small.to_csv(config.TRAIN_METADATA_PATH, index=False)
    val_small.to_csv(config.VAL_METADATA_PATH, index=False)
    test_small.to_csv(config.TEST_METADATA_PATH, index=False)

    print(
        f"    Created small datasets: Train={len(train_small)}, Val={len(val_small)}, Test={len(test_small)}"
    )

    # ------------------------------------------------------------------------
    # 3. Verify Utilities
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Utilities...")

    # Test MetricMonitor
    monitor = MetricMonitor()
    monitor.update("Loss", 0.5)
    monitor.update("Loss", 0.3)
    assert monitor.metrics["Loss"]["count"] == 2
    assert abs(monitor.metrics["Loss"]["avg"] - 0.4) < 1e-6
    print("    MetricMonitor check passed.")

    # Test Mixup
    # Create dummy batch: Batch Size 4, 1 Channel, 32 Freq, 32 Time
    dummy_x = torch.randn(4, 1, 32, 32).to(config.DEVICE)
    dummy_y = torch.tensor([0, 1, 2, 3]).to(config.DEVICE)

    mixed_x, y_a, y_b, lam = mixup_data(
        dummy_x, dummy_y, alpha=1.0, device=config.DEVICE
    )

    assert mixed_x.shape == dummy_x.shape
    assert y_a.shape == dummy_y.shape
    assert y_b.shape == dummy_y.shape
    assert 0.0 <= lam <= 1.0
    print("    Mixup logic check passed.")

    # ------------------------------------------------------------------------
    # 4. Verify Dataset & DataLoader
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Dataset...")

    # Test direct instantiation
    ds_val = SpeechCommandDataset(val_small, mode="val", config=config)
    spec, label = ds_val[0]

    # Check shapes
    # Expected Spectrogram shape: (1, N_MELS, TimeFrames)
    # TimeFrames = (SampleRate * Duration) // HopLength + 1 approx
    # 16000 * 1.0 / 160 = 100 frames -> +1 = 101
    expected_shape = (1, config.N_MELS, 101)

    assert (
        spec.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {spec.shape}"
    assert isinstance(label, torch.Tensor)
    print(f"    __getitem__ check passed. Output shape: {spec.shape}")

    # Test Factory Function (get_dataset) with balancing logic
    # This will generate 'train_balanced.parquet' in the demo working dir
    ds_train = get_dataset("train", config=config, load_cached_data=False)
    print(f"    get_dataset('train') passed. Balanced size: {len(ds_train)}")

    # Create Loaders
    train_loader = DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=config.BATCH_SIZE, shuffle=False)

    # Check batch loading
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (config.BATCH_SIZE, 1, config.N_MELS, 101)
    assert batch_y.shape == (config.BATCH_SIZE,)
    print("    DataLoader batch check passed.")

    # ------------------------------------------------------------------------
    # 5. Verify Model
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Model...")

    model = get_model(config).to(config.DEVICE)

    # Forward pass with the batch we just loaded
    batch_x = batch_x.to(config.DEVICE)
    output = model(batch_x)

    # Check output shape: (Batch, NumClasses)
    assert output.shape == (config.BATCH_SIZE, config.NUM_CLASSES)
    print(f"    Forward pass passed. Output shape: {output.shape}")

    # ------------------------------------------------------------------------
    # 6. Verify Training Loop
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Run one epoch of training
    print("    Running train_one_epoch...")
    train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, config.DEVICE, config
    )
    print(f"    Train Metrics: {train_metrics}")

    # Run validation
    print("    Running validation...")
    val_metrics = validate(model, val_loader, criterion, config.DEVICE, config)
    print(f"    Val Metrics: {val_metrics}")

    # Save model (simulating the save in the main loop)
    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
    assert os.path.exists(config.BEST_MODEL_PATH)
    print("    Model checkpoint saving passed.")

    # ------------------------------------------------------------------------
    # 7. Verify Inference / Submission
    # ------------------------------------------------------------------------
    print("\n[7] Verifying Inference...")

    # Load the saved model
    loaded_model = get_model(config).to(config.DEVICE)
    loaded_model.load_state_dict(
        torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    )
    loaded_model.eval()

    # Setup Test Loader
    ds_test = get_dataset("test", config=config)
    test_loader = DataLoader(ds_test, batch_size=config.BATCH_SIZE, shuffle=False)

    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.DEVICE)
            output = loaded_model(data)
            _, preds = torch.max(output, 1)
            predictions.extend(preds.cpu().numpy())

    # Check prediction count
    assert len(predictions) == len(ds_test)

    # Map back to labels
    pred_labels = [config.ID2LABEL[p] for p in predictions]

    # Create submission dataframe
    fnames = ds_test.df["filepath"].apply(os.path.basename).tolist()
    df_sub = pd.DataFrame({"fname": fnames, "label": pred_labels})

    # Save
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    assert os.path.exists(config.SUBMISSION_PATH)
    print(f"    Submission generated at {config.SUBMISSION_PATH}")
    print(f"    First few predictions:\n{df_sub.head()}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
