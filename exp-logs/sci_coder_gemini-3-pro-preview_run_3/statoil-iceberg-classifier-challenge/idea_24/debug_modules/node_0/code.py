import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

# Import provided library modules
from library.utils import load_data, set_seed
from library.dataset import IcebergDataset
from library.model import CAMA_CNN, train_one_epoch, validate
from library.engine import run


def demonstrate_data_loading(cache_dir):
    print("\n=== Demonstrating Data Loading ===")
    # Load data using the utility function
    # This processes train.json and test.json and caches them as .npy files
    data = load_data(cache_dir=cache_dir, load_cached_data=False)

    (
        X_train,
        y_train,
        angles_train,
        X_val,
        y_val,
        angles_val,
        X_test,
        ids_test,
        angles_test,
    ) = data

    # Validate Train Data
    print(f"Train Data Shape: {X_train.shape}")
    assert len(X_train) > 0, "X_train should not be empty"
    assert X_train.ndim == 4, "X_train should be 4D (N, C, H, W)"
    assert X_train.shape[1] == 3, "Images should have 3 channels (Band1, Band2, Avg)"
    assert X_train.shape[2:] == (75, 75), "Images should be 75x75"
    assert len(y_train) == len(X_train), "y_train length mismatch"
    assert len(angles_train) == len(X_train), "angles_train length mismatch"

    # Validate Test Data
    print(f"Test Data Shape: {X_test.shape}")
    assert len(X_test) > 0, "X_test should not be empty"
    assert len(ids_test) == len(X_test), "ids_test length mismatch"

    return X_train, y_train, angles_train


def demonstrate_dataset(X, y, angles):
    print("\n=== Demonstrating Dataset and DataLoader ===")
    # Create a small subset for demonstration
    subset_size = 16
    X_sub = X[:subset_size]
    y_sub = y[:subset_size]
    angles_sub = angles[:subset_size]

    # Instantiate Dataset
    dataset = IcebergDataset(X_sub, angles_sub, y_sub, transform=None)

    # Instantiate DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Fetch one batch
    images, batch_angles, labels = next(iter(loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Angles Shape: {batch_angles.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (4, 3, 75, 75), "Incorrect batch image shape"
    assert batch_angles.shape == (4,), "Incorrect batch angle shape"
    assert labels.shape == (4,), "Incorrect batch label shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"


def demonstrate_model_logic():
    print("\n=== Demonstrating Model Architecture (CAMA_CNN) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAMA_CNN().to(device)

    # Create dummy input
    batch_size = 8
    dummy_images = torch.randn(batch_size, 3, 75, 75).to(device)
    dummy_angles = torch.randn(batch_size).to(device)

    # Forward pass
    output = model(dummy_images, dummy_angles)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (batch_size, 1), "Output shape should be (Batch_Size, 1)"
    # Check that output is not NaN
    assert not torch.isnan(output).any(), "Model output contains NaNs"


def demonstrate_training_step(X, y, angles):
    print("\n=== Demonstrating Training and Validation Step ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare small data
    dataset = IcebergDataset(X[:32], angles[:32], y[:32])
    loader = DataLoader(dataset, batch_size=8)

    model = CAMA_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train one epoch
    initial_loss = validate(model, loader, criterion, device)[0]
    train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
    final_loss, preds, targets = validate(model, loader, criterion, device)

    print(f"Initial Val Loss: {initial_loss:.4f}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Final Val Loss: {final_loss:.4f}")

    # Assertions
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert len(preds) == 32, "Predictions length mismatch"
    assert len(targets) == 32, "Targets length mismatch"
    # Loss should ideally decrease or change, but strictly checking it's finite is enough for a demo
    assert np.isfinite(train_loss), "Train loss is not finite"


def demonstrate_full_engine(cache_dir, submission_path):
    print("\n=== Demonstrating Full Engine Execution ===")

    # Run the engine with minimal parameters for speed
    # n_folds=2 and epochs=1 ensures we just check the loop mechanics without long wait
    run(
        batch_size=16,
        epochs=1,
        patience=1,
        lr=1e-3,
        weight_decay=1e-4,
        n_folds=2,
        seed=42,
        cache_dir=cache_dir,
        submission_path=submission_path,
    )

    # Verify submission file creation
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission File Rows: {len(df_sub)}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "is_iceberg" in df_sub.columns, "Submission missing 'is_iceberg' column"
    assert len(df_sub) > 0, "Submission file is empty"


if __name__ == "__main__":
    # Configuration
    set_seed(42)
    WORKING_DIR = "./working/demo_usage"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Clean up previous run if exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # 1. Data Loading
        X_train, y_train, angles_train = demonstrate_data_loading(CACHE_DIR)

        # 2. Dataset Logic
        demonstrate_dataset(X_train, y_train, angles_train)

        # 3. Model Logic
        demonstrate_model_logic()

        # 4. Training Step Logic
        demonstrate_training_step(X_train, y_train, angles_train)

        # 5. Full Engine Logic
        demonstrate_full_engine(CACHE_DIR, SUBMISSION_PATH)

        print("\nAll demonstrations completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        raise e
