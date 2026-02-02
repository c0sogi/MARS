import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, get_cached_data
from library.dataset import get_dataloaders
from library.model import SpatialSiameseEfficientNet
from library.trainer import run_training, probabilistic_f1

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1  # Note: trainer.py forces 2 epochs in debug mode
    Config.BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_run"

    # Update derived paths
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Clean and recreate working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Probabilistic F1 Score
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    score = probabilistic_f1(y_true, y_pred)
    print(f"    pF1 Score (Test): {score:.4f}")

    assert 0.0 <= score <= 1.0, "pF1 score is out of valid range [0, 1]"
    assert score > 0.8, "pF1 score calculation seems incorrect for good predictions"

    # Test Caching Mechanism
    print("    Testing caching mechanism...")
    cache_file = os.path.join(Config.WORKING_DIR, "test_cache.parquet")

    def dummy_data_gen():
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # First call: Generate and save
    df_gen = get_cached_data(cache_file, dummy_data_gen, load_cached=False)
    assert os.path.exists(cache_file), "Cache file was not created."

    # Second call: Load from cache
    df_cached = get_cached_data(cache_file, dummy_data_gen, load_cached=True)
    pd.testing.assert_frame_equal(df_gen, df_cached)
    print("    Caching mechanism verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Get dataloaders (Debug mode loads a small subset)
    # load_cached=False ensures we test the processing logic
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached=False, debug=True
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    # Fetch a single batch
    batch = next(iter(train_loader))
    images = batch["image"]
    contra_images = batch["contra_image"]
    labels = batch["label"]

    print(f"    Batch Keys: {list(batch.keys())}")
    print(f"    Image Shape: {images.shape}")
    print(f"    Label Shape: {labels.shape}")

    # Assertions for shapes
    # Expected: (Batch, Channels=3, Height=768, Width=768)
    expected_shape = (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )

    assert (
        images.shape == expected_shape
    ), f"Image tensor shape mismatch. Expected {expected_shape}, got {images.shape}"
    assert (
        contra_images.shape == expected_shape
    ), "Contralateral image tensor shape mismatch."
    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {labels.shape}"

    # Check value ranges (Images should be normalized 0-1 roughly, but standardized age is float)
    # Just checking they are tensors
    assert isinstance(images, torch.Tensor)

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = SpatialSiameseEfficientNet()
    model.to(Config.DEVICE)
    model.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model Parameters: {params:,}")

    # Forward pass verification
    with torch.no_grad():
        img_dev = images.to(Config.DEVICE)
        contra_dev = contra_images.to(Config.DEVICE)

        logits = model(img_dev, contra_dev)

    print(f"    Output Logits Shape: {logits.shape}")

    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch."
    assert not torch.isnan(logits).any(), "Model output contains NaNs."

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Loop...")
    print("    Running short training session (Debug Mode)...")

    # Run training
    # This will run for 2 epochs (hardcoded for debug in trainer.py) on the subset
    run_training(debug=True, load_cached=True)

    # Verify artifact generation
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"    Success: Best model saved at {Config.BEST_MODEL_PATH}")
        file_size = os.path.getsize(Config.BEST_MODEL_PATH) / (1024 * 1024)
        print(f"    Model Size: {file_size:.2f} MB")
    else:
        raise FileNotFoundError("Training completed but best_model.pth was not found.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
