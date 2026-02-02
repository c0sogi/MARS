import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import provided library modules
import library.config as config
import library.utils as utils
from library.dataset import HMSDataset
from library.model import HybridModel
from library.train import train
from library.inference import predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo.
    Creates a small subset of the metadata to ensure the code runs quickly.
    """
    print("\n[1] Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config Paths to point to this demo directory
    config.WORKING_DIR = demo_dir
    config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override Hyperparameters for speed
    config.EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Create Subset Metadata (Train)
    # We take the first 20 rows to make dataset processing instant
    full_train_df = pd.read_csv(config.TRAIN_META_PATH)
    subset_train_df = full_train_df.head(20).copy()
    demo_train_path = os.path.join(demo_dir, "train_subset.csv")
    subset_train_df.to_csv(demo_train_path, index=False)
    config.TRAIN_META_PATH = demo_train_path
    print(f"    Created subset train metadata: {demo_train_path} (20 rows)")

    # Create Subset Metadata (Val)
    full_val_df = pd.read_csv(config.VAL_META_PATH)
    subset_val_df = full_val_df.head(10).copy()
    demo_val_path = os.path.join(demo_dir, "val_subset.csv")
    subset_val_df.to_csv(demo_val_path, index=False)
    config.VAL_META_PATH = demo_val_path
    print(f"    Created subset val metadata: {demo_val_path} (10 rows)")

    # Create Subset Metadata (Test)
    full_test_df = pd.read_csv(config.TEST_META_PATH)
    subset_test_df = full_test_df.head(10).copy()
    demo_test_path = os.path.join(demo_dir, "test_subset.csv")
    subset_test_df.to_csv(demo_test_path, index=False)
    config.TEST_META_PATH = demo_test_path
    print(f"    Created subset test metadata: {demo_test_path} (10 rows)")


def verify_dataset_logic():
    """
    Instantiates the dataset and verifies output shapes and types.
    """
    print("\n[2] Verifying HMSDataset Logic...")

    # Initialize Dataset (this will trigger processing and caching for the subset)
    # We use load_cached_data=False to force the processing logic to run
    ds = HMSDataset(
        csv_file=config.TRAIN_META_PATH,
        mode="train",
        augment=False,
        load_cached_data=False,
    )

    print(f"    Dataset length: {len(ds)}")
    assert len(ds) == 20, "Dataset length mismatch with subset metadata."

    # Fetch one sample
    eeg, spec, target = ds[0]

    # Verify EEG Shape: (19, 2500) -> (Channels, Time)
    print(f"    EEG Tensor Shape: {eeg.shape}")
    assert eeg.shape == (19, 2500), f"Expected EEG shape (19, 2500), got {eeg.shape}"
    assert isinstance(eeg, torch.Tensor), "EEG data is not a Tensor"

    # Verify Spectrogram Shape: (4, 256, 256) -> (Channels, Height, Width)
    print(f"    Spec Tensor Shape: {spec.shape}")
    assert spec.shape == (
        4,
        256,
        256,
    ), f"Expected Spec shape (4, 256, 256), got {spec.shape}"

    # Verify Target Shape: (6,)
    print(f"    Target Tensor Shape: {target.shape}")
    assert target.shape == (6,), f"Expected Target shape (6,), got {target.shape}"

    # Verify Target Sum (should be approx 1.0 for probabilities)
    target_sum = target.sum().item()
    assert (
        abs(target_sum - 1.0) < 1e-4
    ), f"Target probabilities do not sum to 1. Sum={target_sum}"

    print("    Dataset verification passed.")


def verify_model_logic():
    """
    Instantiates the model and runs a forward pass with dummy data.
    """
    print("\n[3] Verifying HybridModel Logic...")

    device = config.DEVICE
    model = HybridModel().to(device)
    model.eval()

    # Create dummy batch
    batch_size = 2
    dummy_eeg = torch.randn(batch_size, 19, 2500).to(device)
    dummy_spec = torch.randn(batch_size, 4, 256, 256).to(device)

    print("    Running forward pass on dummy data...")
    with torch.no_grad():
        logits = model(dummy_eeg, dummy_spec)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    expected_shape = (batch_size, config.NUM_CLASSES)
    assert (
        logits.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("    Model verification passed.")


def run_training_demo():
    """
    Runs the training loop using the subset data.
    """
    print("\n[4] Running Training Demo...")

    # We set debug=False here because we manually created the subsets in step 1.
    # If we used debug=True, it would try to subset the already small dataset again.
    # load_cached_data=True because verify_dataset_logic already created the cache.
    train(debug=False, load_cached_data=True)

    # Verify checkpoint creation
    if os.path.exists(config.MODEL_PATH):
        print(f"    Checkpoint successfully saved at {config.MODEL_PATH}")
    else:
        raise FileNotFoundError(
            "Training completed but model checkpoint was not found."
        )


def run_inference_demo():
    """
    Runs inference using the trained model and verifies submission output.
    """
    print("\n[5] Running Inference Demo...")

    submission_df = predict(load_cached_data=False)

    # Verify Submission
    print("    Verifying submission file...")
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    # Check columns
    expected_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {submission_df.columns}"

    # Check row count
    assert (
        len(submission_df) == 10
    ), f"Expected 10 predictions (subset size), got {len(submission_df)}"

    # Check probability sum
    vote_cols = expected_cols[1:]
    row_sums = submission_df[vote_cols].sum(axis=1)
    # Allow small float error
    valid_sums = np.allclose(row_sums, 1.0, atol=1e-4)
    assert valid_sums, "Predicted probabilities do not sum to 1.0"

    print("    Inference verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    try:
        setup_demo_environment()
        verify_dataset_logic()
        verify_model_logic()
        run_training_demo()
        run_inference_demo()
        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        raise e
