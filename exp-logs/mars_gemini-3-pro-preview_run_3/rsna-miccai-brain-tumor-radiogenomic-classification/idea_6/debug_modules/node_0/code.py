import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import (
    set_seed,
    read_dicom_file,
    normalize_min_max,
    get_depth_indices,
)
from library.data import prepare_datasets, BraTSDataset
from library.model import DualStreamSiameseNet
from library.train import run_training


def create_subset_metadata(
    source_dir, target_dir, train_size=12, val_size=4, test_size=4
):
    """
    Creates a small subset of the metadata files to speed up the demonstration.
    """
    os.makedirs(target_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_parquet(os.path.join(source_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(source_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(source_dir, "test.parquet"))

    # Subset
    train_subset = train_df.head(train_size)
    val_subset = val_df.head(val_size)
    test_subset = test_df.head(test_size)

    # Save to new location
    train_subset.to_parquet(os.path.join(target_dir, "train.parquet"), index=False)
    val_subset.to_parquet(os.path.join(target_dir, "val.parquet"), index=False)
    test_subset.to_parquet(os.path.join(target_dir, "test.parquet"), index=False)

    print(f"Created metadata subset in {target_dir}")
    print(
        f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )
    return train_subset


def verify_utils(sample_path):
    """
    Verifies utility functions.
    """
    print("\n[1] Verifying Utils...")

    # 1. Test read_dicom_file
    # Note: sample_path is relative to ./input
    img = read_dicom_file(sample_path)

    assert isinstance(img, np.ndarray), "read_dicom_file should return a numpy array"
    assert img.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected ({Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    print(f" - read_dicom_file: Success. Shape {img.shape}")

    # 2. Test normalize_min_max
    dummy_vol = np.array([-100, 0, 100, 200], dtype=np.float32)
    norm_vol = normalize_min_max(dummy_vol)
    assert np.isclose(norm_vol.min(), 0.0) and np.isclose(
        norm_vol.max(), 1.0
    ), "Normalization failed to scale to [0, 1]"
    print(" - normalize_min_max: Success.")

    # 3. Test get_depth_indices
    # Request 32 slices from a volume of 100 slices
    indices = get_depth_indices(total_slices=100, num_target_slices=32)
    assert len(indices) == 32, "Incorrect number of indices returned"
    assert indices.min() >= 10, "Indices should start after 10% depth"
    assert indices.max() < 90, "Indices should end before 90% depth"
    print(" - get_depth_indices: Success.")


def verify_data_pipeline():
    """
    Verifies data loading and processing.
    """
    print("\n[2] Verifying Data Pipeline...")

    # Run prepare_datasets with load_cached_data=False to force processing of our subset
    # This uses the Config.METADATA_DIR which we will override in main
    train_ds, val_ds, test_ds = prepare_datasets(load_cached_data=False, num_workers=2)

    assert len(train_ds) > 0, "Train dataset is empty"

    # Fetch one item
    x_even, x_odd, label = train_ds[0]

    # Check shapes
    # Expected: (64, 256, 256) for each stream
    # 64 channels = 16 slices * 4 modalities
    expected_shape = (Config.INPUT_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)

    assert x_even.shape == expected_shape, f"Even stream shape mismatch: {x_even.shape}"
    assert x_odd.shape == expected_shape, f"Odd stream shape mismatch: {x_odd.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    print(f" - Data Processing: Success. Tensor Shape: {x_even.shape}")
    return train_ds, val_ds, test_ds


def verify_model():
    """
    Verifies model architecture and forward pass.
    """
    print("\n[3] Verifying Model...")

    model = DualStreamSiameseNet()
    model.eval()

    # Create dummy input batch: (Batch=2, Channels=64, H=256, W=256)
    B = 2
    dummy_even = torch.randn(B, Config.INPUT_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)
    dummy_odd = torch.randn(B, Config.INPUT_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)

    with torch.no_grad():
        output = model(dummy_even, dummy_odd)

    assert output.shape == (
        B,
        1,
    ), f"Model output shape mismatch. Expected ({B}, 1), got {output.shape}"
    print(f" - Model Forward Pass: Success. Output Shape: {output.shape}")
    return model


def run_demo_training():
    """
    Runs the training loop using the library function.
    """
    print("\n[4] Running Training Loop...")

    # Config has already been modified in main()
    trained_model = run_training(
        load_cached_data=True,  # Use the cache we just generated in verify_data_pipeline
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
    )

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(" - Training Loop: Success. Model saved.")
    return trained_model


def generate_submission(model, test_dataset):
    """
    Generates a submission file using the trained model.
    """
    print("\n[5] Generating Submission...")

    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    predictions = []
    ids = []

    with torch.no_grad():
        for x_even, x_odd, patient_ids in test_loader:
            x_even = x_even.to(device)
            x_odd = x_odd.to(device)

            logits = model(x_even, x_odd)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            ids.extend(patient_ids)

    # Create DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Save
    output_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f" - Submission generated at {output_path}")
    print(submission_df.head())

    assert len(submission_df) == len(test_dataset), "Submission row count mismatch"


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 0. Setup and Overrides for Demo
    # ------------------------------------------------------------------
    set_seed(42)

    # Define demo directories
    DEMO_DIR = "./working/demo_workspace"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")

    # Override Config class attributes for this run
    Config.WORKING_DIR = DEMO_DIR
    Config.METADATA_DIR = DEMO_META_DIR
    Config.NUM_EPOCHS = 2  # Reduce epochs for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Create Data Subset
    # ------------------------------------------------------------------
    # We use the original metadata to create a small subset for the demo
    train_subset_df = create_subset_metadata(
        source_dir="./metadata", target_dir=Config.METADATA_DIR
    )

    # ------------------------------------------------------------------
    # 2. Verify Utils
    # ------------------------------------------------------------------
    # Get a valid file path from the subset dataframe for testing
    sample_flair_paths = train_subset_df.iloc[0]["flair_paths"]
    if len(sample_flair_paths) > 0:
        verify_utils(sample_flair_paths[0])
    else:
        print(
            "Warning: No flair paths found in first sample, skipping utils verification for file read."
        )

    # ------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # ------------------------------------------------------------------
    # This will process the subset metadata and cache numpy arrays in Config.WORKING_DIR
    train_ds, val_ds, test_ds = verify_data_pipeline()

    # ------------------------------------------------------------------
    # 4. Verify Model
    # ------------------------------------------------------------------
    verify_model()

    # ------------------------------------------------------------------
    # 5. Run Training
    # ------------------------------------------------------------------
    trained_model = run_demo_training()

    # ------------------------------------------------------------------
    # 6. Generate Submission
    # ------------------------------------------------------------------
    generate_submission(trained_model, test_ds)

    print("\n" + "=" * 40)
    print(" DEMONSTRATION COMPLETE")
    print("=" * 40)
