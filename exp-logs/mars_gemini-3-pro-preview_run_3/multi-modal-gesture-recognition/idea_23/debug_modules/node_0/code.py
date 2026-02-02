import os
import sys
import shutil
import random
import numpy as np
import pandas as pd
import torch

# Import library modules
from library import config
from library import data_utils
from library import features
from library import dataset
from library import model
from library import loss
from library import trainer
from library import inference


def setup_demo_environment():
    """
    Overrides config parameters for a quick demo run and creates subset metadata.
    """
    print("Setting up demo environment...")

    # 1. Define Working Paths
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Override Config Parameters for Speed
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 4
    config.HIDDEN_DIM = 32  # Reduced capacity for speed
    config.ENCODER_LAYERS = 1
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Override Paths
    config.CACHE_DIR = os.path.join(demo_dir, "cache")
    config.MODEL_SAVE_PATH = os.path.join(demo_dir, "mstcn_demo_model.pth")
    config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # 3. Create Subset Metadata (Mini-Dataset)
    # We read the original metadata and take the top N rows to create a fast pipeline
    subset_size = 10

    meta_files = {
        "train": (config.TRAIN_METADATA_PATH, "train_subset.csv"),
        "val": (config.VAL_METADATA_PATH, "val_subset.csv"),
        "test": (config.TEST_METADATA_PATH, "test_subset.csv"),
    }

    new_paths = {}

    for key, (src_path, filename) in meta_files.items():
        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Take a subset
            df_subset = df.head(subset_size)
            dest_path = os.path.join(demo_dir, filename)
            df_subset.to_csv(dest_path, index=False)
            new_paths[key] = dest_path
            print(f"Created subset for {key}: {len(df_subset)} samples -> {dest_path}")
        else:
            raise FileNotFoundError(f"Original metadata not found at {src_path}")

    # Update Config to point to subsets
    config.TRAIN_METADATA_PATH = new_paths["train"]
    config.VAL_METADATA_PATH = new_paths["val"]
    config.TEST_METADATA_PATH = new_paths["test"]


def test_data_loading():
    print("\n=== Testing Data Loading & Processing ===")

    # Load data using the library function (this will use the subset metadata)
    train_data, val_data, test_data, stats = features.load_data_and_stats(
        load_cached_data=False
    )

    # Assertions
    assert len(train_data["skeleton"]) > 0, "Train data skeleton list is empty"
    assert len(train_data["audio"]) == len(
        train_data["skeleton"]
    ), "Audio/Skeleton mismatch"
    assert "kinematics_mean" in stats, "Stats missing kinematics_mean"
    assert "audio_mean" in stats, "Stats missing audio_mean"

    print("Data loaded successfully.")
    print(f"Train samples: {len(train_data['sample_ids'])}")
    print(f"Stats Kinematics Mean Shape: {stats['kinematics_mean'].shape}")

    return train_data, val_data, test_data, stats


def test_dataset_and_loader():
    print("\n=== Testing Dataset & DataLoader ===")

    # Create DataLoaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Fetch one batch
    feat, target, sample_ids, start_frames = next(iter(train_loader))

    print(f"Batch Feature Shape: {feat.shape}")
    print(f"Batch Target Shape: {target.shape}")

    # Assertions
    # Feature shape: (Batch, WindowSize, InputDim)
    assert feat.dim() == 3, f"Expected 3D feature tensor, got {feat.dim()}"
    assert (
        feat.shape[1] == config.WINDOW_SIZE
    ), f"Expected window size {config.WINDOW_SIZE}, got {feat.shape[1]}"
    assert (
        feat.shape[2] == config.INPUT_DIM
    ), f"Expected input dim {config.INPUT_DIM}, got {feat.shape[2]}"

    # Target shape: (Batch, WindowSize)
    assert target.dim() == 2, f"Expected 2D target tensor, got {target.dim()}"
    assert target.shape[1] == config.WINDOW_SIZE, "Target time dimension mismatch"

    return train_loader, val_loader


def test_model_logic(train_loader):
    print("\n=== Testing Model Architecture ===")

    device = torch.device("cpu")  # Use CPU for simple logic check

    # Instantiate Model
    weskn = model.WESKN().to(device)

    # Get a batch
    feat, target, _, _ = next(iter(train_loader))
    feat = feat.to(device)
    target = target.to(device)

    # Forward Pass
    outputs = weskn(feat)

    # Check outputs
    required_keys = ["logits_1", "logits_2", "logits_3", "probs_3"]
    for k in required_keys:
        assert k in outputs, f"Model output missing key: {k}"

    # Check shape of logits: (Batch, Time, NumClasses)
    logits = outputs["logits_3"]
    assert logits.shape == (
        feat.shape[0],
        config.WINDOW_SIZE,
        config.NUM_CLASSES,
    ), f"Logits shape mismatch: {logits.shape}"

    print("Model forward pass successful. Output shapes verified.")

    return weskn, outputs, target


def test_loss_function(outputs, target):
    print("\n=== Testing Loss Function ===")

    criterion = loss.CascadedSmoothnessLoss()

    # Calculate loss
    loss_val = criterion(outputs, target)

    print(f"Calculated Loss: {loss_val.item()}")

    # Assertions
    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val > 0, "Loss should be positive"
    assert loss_val.dim() == 0, "Loss should be a scalar"


def test_training_loop(train_loader, val_loader):
    print("\n=== Testing Training Loop ===")

    # We will run the actual trainer function which handles the loop
    # It uses the config we overrode earlier (2 epochs)
    trained_model = trainer.train_model(train_loader, val_loader)

    # Verify model file was saved
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), f"Model file not found at {config.MODEL_SAVE_PATH}"
    print(f"Training complete. Model saved to {config.MODEL_SAVE_PATH}")


def test_inference():
    print("\n=== Testing Inference & Submission ===")

    # Run inference generation
    # This uses the saved model from the previous step and the test subset
    inference.generate_submission(load_cached_data=True)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    # Check content format
    with open(config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    assert len(lines) > 0, "Submission file is empty"

    # Check first line format: SessionID,Labels...
    # Example: Session00001,2,12,3
    sample_line = lines[0].strip()
    parts = sample_line.split(",")

    print(f"Sample Submission Line: {sample_line}")
    assert len(parts) >= 1, "Invalid submission line format"
    # Note: It's possible to have no predictions (just SessionID), so len >= 1 is the minimal check

    print("Inference successful.")


if __name__ == "__main__":
    # 1. Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        # 2. Setup
        setup_demo_environment()

        # 3. Data Loading
        train_data, val_data, test_data, stats = test_data_loading()

        # 4. Dataset & Loader
        train_loader, val_loader = test_dataset_and_loader()

        # 5. Model Logic
        weskn_model, outputs, target = test_model_logic(train_loader)

        # 6. Loss Function
        test_loss_function(outputs, target)

        # 7. Training Loop
        test_training_loop(train_loader, val_loader)

        # 8. Inference
        test_inference()

        print("\nAll tests passed successfully!")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
