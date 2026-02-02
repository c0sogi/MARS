import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil

# Import provided library modules
import library.config as config
import library.data_utils as data_utils
import library.dataset as dataset
import library.model as model
import library.train_utils as train_utils


def setup_environment():
    """Sets random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create a directory for demo files
    os.makedirs(config.WORKING_DIR, exist_ok=True)


def create_demo_data_subsets():
    """
    Creates small subsets of metadata and sample submission to ensure
    the demo runs quickly.
    """
    print("Creating data subsets for rapid demonstration...")

    subset_size = 500  # Small number of rows for speed

    # 1. Create Train Metadata Subset
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH, nrows=subset_size)
    subset_train_path = os.path.join(config.WORKING_DIR, "train_subset.csv")
    df_train.to_csv(subset_train_path, index=False)

    # 2. Create Validation Metadata Subset
    df_val = pd.read_csv(config.VAL_METADATA_PATH, nrows=subset_size)
    subset_val_path = os.path.join(config.WORKING_DIR, "val_subset.csv")
    df_val.to_csv(subset_val_path, index=False)

    # 3. Create Test Metadata Subset
    df_test = pd.read_csv(config.TEST_METADATA_PATH, nrows=subset_size)
    subset_test_path = os.path.join(config.WORKING_DIR, "test_subset.csv")
    df_test.to_csv(subset_test_path, index=False)

    # 4. Create Sample Submission Subset
    # The submission file must match the test metadata length/order
    df_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH, nrows=subset_size)
    subset_sub_path = os.path.join(config.WORKING_DIR, "sample_submission_subset.csv")
    df_sub.to_csv(subset_sub_path, index=False)

    # 5. Patch Configuration to use these subsets
    # We modify the module-level variables in config so other modules use them
    config.TRAIN_METADATA_PATH = subset_train_path
    config.VAL_METADATA_PATH = subset_val_path
    config.TEST_METADATA_PATH = subset_test_path
    config.SAMPLE_SUBMISSION_PATH = subset_sub_path

    # Also patch cache paths to avoid overwriting real caches or loading old large ones
    config.TRAIN_FEATURES_PATH = os.path.join(
        config.WORKING_DIR, "train_features_subset.parquet"
    )
    config.VAL_FEATURES_PATH = os.path.join(
        config.WORKING_DIR, "val_features_subset.parquet"
    )
    config.TEST_FEATURES_PATH = os.path.join(
        config.WORKING_DIR, "test_features_subset.parquet"
    )

    print("Subsets created and configuration patched.")


def demonstrate_data_processing():
    """
    Demonstrates loading, processing, and scaling of data.
    """
    print("\n--- Data Processing ---")

    # Load and process data (this uses the patched config paths)
    # We set load_cached_data=False to force processing of our new subsets
    df_train_wide = data_utils.load_and_process_data(
        split="train", load_cached_data=False
    )
    df_val_wide = data_utils.load_and_process_data(split="val", load_cached_data=False)
    df_test_wide = data_utils.load_and_process_data(
        split="test", load_cached_data=False
    )

    # Verify processing results
    print(f"Processed Train Shape: {df_train_wide.shape}")
    assert not df_train_wide.empty, "Train dataframe is empty"
    assert "is_ground" in df_train_wide.columns, "is_ground column missing"

    # Scale data
    X_train, X_val, X_test, scaler = data_utils.scale_data(
        df_train_wide, df_val_wide, df_test_wide
    )

    # Verify scaling
    assert (
        X_train.shape[1] == config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {config.INPUT_DIM}, got {X_train.shape[1]}"
    assert np.abs(X_train.mean()) < 0.1, "Scaled data mean should be close to 0"

    # Extract targets and ground indicators
    y_train = df_train_wide["contact"].values
    g_train = df_train_wide["is_ground"].values

    y_val = df_val_wide["contact"].values
    g_val = df_val_wide["is_ground"].values

    # Test set has no targets in reality, but for code consistency we might generate dummies or handle it
    # The dataset class handles None targets.
    g_test = df_test_wide["is_ground"].values

    return (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, None, g_test)


def demonstrate_dataset_and_loader(train_data):
    """
    Demonstrates dataset instantiation and DataLoader usage.
    """
    print("\n--- Dataset & DataLoader ---")
    X, y, g = train_data

    # Instantiate Dataset
    ds = dataset.ContactDataset(features=X, targets=y, is_ground=g)

    # Verify single item retrieval
    x_sample, y_sample, g_sample = ds[0]
    assert isinstance(x_sample, torch.Tensor), "Feature should be a tensor"
    assert x_sample.shape[0] == config.INPUT_DIM, "Incorrect feature dimension"
    assert isinstance(g_sample, torch.Tensor), "Ground indicator should be a tensor"

    print(f"Dataset size: {len(ds)}")
    print(f"Sample feature shape: {x_sample.shape}")

    # Create DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    # Verify batch
    batch_x, batch_y, batch_g = next(iter(loader))
    assert batch_x.shape[0] == 32, "Batch size mismatch"
    assert batch_x.shape[1] == config.INPUT_DIM, "Batch feature dim mismatch"

    return loader


def demonstrate_model(loader):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n--- Model Initialization ---")

    # Instantiate Model
    # Using default parameters from config/class definition
    net = model.WITRGN(
        num_features=config.NUM_FEATURES_PER_STEP, window_size=config.WINDOW_SIZE
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Get a batch for testing
    batch_x, batch_y, batch_g = next(iter(loader))
    batch_x = batch_x.to(device)
    batch_g = batch_g.to(device)

    # Forward pass
    logits = net(batch_x, batch_g)

    # Verify output
    assert logits.shape == (
        32,
        1,
    ), f"Output shape mismatch. Expected (32, 1), got {logits.shape}"
    print("Model forward pass successful.")

    return net


def demonstrate_training(net, train_loader, val_data):
    """
    Demonstrates the training loop and validation.
    """
    print("\n--- Training Loop ---")

    X_val, y_val, g_val = val_data
    val_ds = dataset.ContactDataset(features=X_val, targets=y_val, is_ground=g_val)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Run training for a few epochs with reduced patience for demo
    trained_model = train_utils.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,  # Reduced for speed
        lr=1e-3,
        patience=1,  # Reduced for speed
        save_path=os.path.join(config.WORKING_DIR, "demo_model.pth"),
    )

    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "demo_model.pth")
    ), "Model file was not saved"

    # Optimize Threshold
    best_threshold = train_utils.optimize_threshold(trained_model, val_loader)
    assert 0.0 < best_threshold < 1.0, "Threshold optimization returned invalid value"

    return trained_model, best_threshold


def demonstrate_inference(model, test_data, threshold):
    """
    Demonstrates inference and submission generation.
    """
    print("\n--- Inference & Submission ---")

    X_test, _, g_test = test_data
    test_ds = dataset.ContactDataset(features=X_test, targets=None, is_ground=g_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    output_csv = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    train_utils.generate_submission(
        model=model,
        test_loader=test_loader,
        threshold=threshold,
        output_path=output_csv,
    )

    # Verify submission
    assert os.path.exists(output_csv), "Submission file not found"
    df_sub = pd.read_csv(output_csv)
    assert "contact_id" in df_sub.columns, "Submission missing contact_id"
    assert "contact" in df_sub.columns, "Submission missing contact"
    assert len(df_sub) == len(
        pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    ), "Submission length mismatch"

    print(f"Submission generated successfully at {output_csv}")
    print(df_sub.head())


if __name__ == "__main__":
    # 1. Setup
    setup_environment()
    create_demo_data_subsets()

    # 2. Process Data
    train_data, val_data, test_data = demonstrate_data_processing()

    # 3. Create Loader
    train_loader = demonstrate_dataset_and_loader(train_data)

    # 4. Initialize Model
    net = demonstrate_model(train_loader)

    # 5. Train
    trained_model, best_thresh = demonstrate_training(net, train_loader, val_data)

    # 6. Inference
    demonstrate_inference(trained_model, test_data, best_thresh)

    print("\nDemo completed successfully.")
