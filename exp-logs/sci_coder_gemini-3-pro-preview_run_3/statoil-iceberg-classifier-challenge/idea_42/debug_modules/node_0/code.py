import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config, set_seed
from library.utils import load_dataset
from library.layers import MaxBlurPool2d, SEModule
from library.data import IcebergDataset, get_dataloaders
from library.model import AAHACNN
from library.train import run_kfold


def main():
    # 1. Setup Configuration for Demo
    print("--- Setting up Configuration ---")
    # Modify Config attributes for this demo run to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_usage"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Set hyperparams for a quick run
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1
    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 4  # Small batch size for demo

    # Create directories
    Config.setup()
    set_seed(Config.SEED)
    print("Configuration setup complete.")

    # 2. Verify Data Loading Utility
    print("\n--- Verifying Data Loading (library.utils) ---")
    # Load training data
    X_train, angles_train, y_train = load_dataset("train", load_cached_data=False)
    print(
        f"Train Data Loaded: X={X_train.shape}, angles={angles_train.shape}, y={y_train.shape}"
    )

    # Assertions
    assert isinstance(X_train, np.ndarray)
    assert X_train.ndim == 4 and X_train.shape[1] == 3 and X_train.shape[2:] == (75, 75)
    assert angles_train.shape[0] == X_train.shape[0]
    assert y_train.shape[0] == X_train.shape[0]

    # Load test data
    X_test, angles_test, ids_test = load_dataset("test", load_cached_data=False)
    print(
        f"Test Data Loaded: X={X_test.shape}, angles={angles_test.shape}, ids={ids_test.shape}"
    )
    assert len(ids_test) == len(X_test)

    # 3. Verify Custom Layers
    print("\n--- Verifying Custom Layers (library.layers) ---")
    device = torch.device("cpu")
    dummy_input = torch.randn(2, 64, 32, 32).to(device)  # Batch=2, C=64, H=32, W=32

    # Test MaxBlurPool2d
    mb_pool = MaxBlurPool2d(in_channels=64, kernel_size=3).to(device)
    out_pool = mb_pool(dummy_input)
    print(f"MaxBlurPool2d Input: {dummy_input.shape} -> Output: {out_pool.shape}")
    # Expect spatial dims to halve (32 -> 16)
    assert out_pool.shape == (2, 64, 16, 16), "MaxBlurPool2d output shape mismatch"

    # Test SEModule
    se_module = SEModule(channels=64, reduction=4).to(device)
    out_se = se_module(dummy_input)
    print(f"SEModule Input: {dummy_input.shape} -> Output: {out_se.shape}")
    # Expect shape to remain unchanged
    assert out_se.shape == dummy_input.shape, "SEModule output shape mismatch"

    # 4. Verify Data Pipeline
    print("\n--- Verifying Data Pipeline (library.data) ---")
    # Get dataloaders (debug mode is enabled in Config)
    loaders = get_dataloaders(batch_size=Config.BATCH_SIZE, debug=True)
    train_loader = loaders["train"]

    # Fetch one batch
    images, angles, labels = next(iter(train_loader))
    print(
        f"Batch Shapes - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert labels.shape == (Config.BATCH_SIZE,)
    assert images.dtype == torch.float32

    # 5. Verify Model Architecture
    print("\n--- Verifying Model Architecture (library.model) ---")
    model = AAHACNN().to(device)

    # Forward pass with batch from loader
    output = model(images.to(device), angles.to(device))
    print(f"Model Output Shape: {output.shape}")

    # Expect (Batch_Size, 1) output (logits)
    assert output.shape == (Config.BATCH_SIZE, 1)

    # Check parameter count (sanity check)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {params}")
    assert params > 0

    # 6. Verify Training Loop & Submission
    print("\n--- Verifying Training Loop (library.train) ---")
    # Run K-Fold with minimal epochs and folds
    print("Running K-Fold execution...")
    run_kfold(n_folds=Config.N_FOLDS, epochs=Config.NUM_EPOCHS, debug=True)

    # Check if submission file exists
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        print(f"Submission file generated at: {submission_path}")
        df_sub = pd.read_csv(submission_path)
        print("Submission Head:")
        print(df_sub.head())

        # Verify format
        assert "id" in df_sub.columns
        assert "is_iceberg" in df_sub.columns
        assert len(df_sub) > 0
        # In debug mode, we sliced the test set, so length will be small (32 or 64 depending on implementation details of debug slicing)
        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created by run_kfold.")

    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
