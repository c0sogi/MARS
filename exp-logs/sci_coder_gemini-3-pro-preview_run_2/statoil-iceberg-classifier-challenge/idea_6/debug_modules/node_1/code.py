import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import from provided library files
from library.utils import set_seed, process_data, IcebergDataset, A2SHN
from library.data_loader import get_kfold_loaders, get_test_loader
from library.train_eval import train_fold
from library.model import run as run_pipeline


def demo_data_processing():
    print("\n=== Demo: Data Processing ===")
    base_dir = "./working/demo_execution"

    # Clean up previous demo runs if they exist to ensure fresh processing
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    # 1. Process Data
    # We use a custom base_dir to separate this demo from the main pipeline artifacts
    print(f"Processing data into {base_dir}...")
    X_train, y_train, inc_train, X_test, inc_test, test_ids = process_data(
        load_cached_data=False, base_dir=base_dir
    )

    # 2. Verify Shapes
    print("Verifying data shapes...")
    # X_train should be (N, 3, 75, 75)
    assert len(X_train.shape) == 4, f"Expected 4D X_train, got {X_train.shape}"
    assert X_train.shape[1] == 3, f"Expected 3 channels, got {X_train.shape[1]}"
    assert (
        X_train.shape[2] == 75 and X_train.shape[3] == 75
    ), "Expected 75x75 spatial dims"

    # y_train should be (N,)
    assert len(y_train.shape) == 1, "Expected 1D y_train"
    assert X_train.shape[0] == y_train.shape[0], "Sample count mismatch between X and y"

    # inc_train should be (N,)
    assert len(inc_train.shape) == 1, "Expected 1D inc_train"

    # 3. Verify Content
    print("Verifying data content...")
    # Check normalization (roughly 0-1 range, though min-max scaling per channel is applied)
    # Since it's min-max scaled per channel in process_data, values should be within [0, 1]
    assert X_train.min() >= 0.0, "Found negative values in processed X_train"
    assert (
        X_train.max() <= 1.00001
    ), "Found values > 1.0 in processed X_train"  # tolerance for float precision

    print("Data processing demo passed.")
    return X_train, y_train, inc_train


def demo_dataset_and_loader(X, y, inc):
    print("\n=== Demo: Dataset and DataLoader ===")

    # 1. Instantiate Dataset
    # Use a small subset
    subset_size = 32
    ds = IcebergDataset(
        X[:subset_size], inc[:subset_size], y[:subset_size], transform=True
    )

    # 2. Verify __getitem__
    img, angle, label = ds[0]
    print(
        f"Sample shapes - Img: {img.shape}, Angle: {angle.shape}, Label: {label.shape}"
    )

    assert img.shape == (3, 75, 75), "Incorrect image tensor shape"
    assert angle.shape == (1,), "Incorrect angle tensor shape"
    assert label.shape == (1,), "Incorrect label tensor shape"
    assert isinstance(img, torch.Tensor), "Image is not a tensor"

    # 3. Verify DataLoader
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    batch_imgs, batch_incs, batch_labels = next(iter(loader))

    print(
        f"Batch shapes - Img: {batch_imgs.shape}, Angle: {batch_incs.shape}, Label: {batch_labels.shape}"
    )
    assert batch_imgs.shape == (8, 3, 75, 75), "Incorrect batch image shape"
    assert batch_incs.shape == (8, 1), "Incorrect batch angle shape"
    assert batch_labels.shape == (8, 1), "Incorrect batch label shape"

    print("Dataset and DataLoader demo passed.")


def demo_model_architecture():
    print("\n=== Demo: Model Architecture (A2SHN) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = A2SHN().to(device)

    # Create dummy inputs
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 75, 75).to(device)
    dummy_inc = torch.randn(batch_size, 1).to(device)

    # Forward pass
    output = model(dummy_img, dummy_inc)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {output.shape}"

    # Check output range (Sigmoid activation -> [0, 1])
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output out of sigmoid range [0, 1]"

    print("Model architecture demo passed.")


def demo_training_step(X, y, inc):
    print("\n=== Demo: Training Step (Single Fold) ===")

    # Use a small subset for speed
    subset_indices = np.arange(100)
    X_sub = X[subset_indices]
    y_sub = y[subset_indices]
    inc_sub = inc[subset_indices]

    # Create loaders manually for this specific test
    # Split 80/20
    split = 80
    train_ds = IcebergDataset(
        X_sub[:split], inc_sub[:split], y_sub[:split], transform=True
    )
    val_ds = IcebergDataset(
        X_sub[split:], inc_sub[split:], y_sub[split:], transform=False
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)

    print("Running train_fold for 1 epoch...")
    # Run for 1 epoch
    model = train_fold(
        train_loader,
        val_loader,
        epochs=1,
        lr=1e-3,
        patience=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    assert isinstance(model, torch.nn.Module), "train_fold did not return a model"
    print("Training step demo passed.")


def demo_full_pipeline():
    print("\n=== Demo: Full Pipeline Execution ===")

    # We run the full pipeline with minimal parameters to ensure end-to-end connectivity.
    # Note: run() uses the default directory ./working/idea_6 internally.

    # Parameters optimized for speed
    epochs = 1
    batch_size = 32
    n_splits = 2  # Minimum splits for CV

    print(f"Running pipeline with epochs={epochs}, n_splits={n_splits}...")
    run_pipeline(epochs=epochs, batch_size=batch_size, n_splits=n_splits, seed=42)

    # Verify submission file creation
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created with {len(df_sub)} rows.")
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of range"

    print("Full pipeline demo passed.")


if __name__ == "__main__":
    # Set global seed
    set_seed(42)

    # 1. Process Data & Verify
    X_train, y_train, inc_train = demo_data_processing()

    # 2. Verify Dataset & Loader
    demo_dataset_and_loader(X_train, y_train, inc_train)

    # 3. Verify Model
    demo_model_architecture()

    # 4. Verify Training Logic
    demo_training_step(X_train, y_train, inc_train)

    # 5. Verify Full Pipeline
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
