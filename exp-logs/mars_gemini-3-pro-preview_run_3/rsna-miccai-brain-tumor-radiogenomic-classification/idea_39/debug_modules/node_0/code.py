import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.model import SSFNet
from library.train import run_training
from library.predict import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_demo_metadata(source_path, dest_path, n_samples=5):
    """
    Creates a small subset of the metadata for demonstration purposes.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_parquet(source_path)
    # Take a small subset
    subset_df = df.head(n_samples).copy()
    subset_df.to_parquet(dest_path, index=False)
    print(f"Created demo metadata at {dest_path} with {len(subset_df)} samples.")
    return len(subset_df)


def verify_model_architecture():
    """
    Instantiates the model and checks input/output shapes.
    """
    print("\n[Demo] Verifying Model Architecture...")
    device = "cpu"  # Use CPU for simple shape check
    model = SSFNet().to(device)
    model.eval()

    # Create dummy input: (Batch=2, Channels=64, H=224, W=224)
    # Config.IN_CHANS is 64
    batch_size = 2
    dummy_even = torch.randn(
        batch_size, Config.IN_CHANS, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(device)
    dummy_odd = torch.randn(
        batch_size, Config.IN_CHANS, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(device)

    with torch.no_grad():
        output = model(dummy_even, dummy_odd)

    # Check output shape: (Batch, 1)
    expected_shape = (batch_size, 1)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model architecture verification passed. Output shape is correct.")


def run_demo():
    # 1. Setup Environment
    print("=" * 40)
    print(" STARTING DEMO EXECUTION")
    print("=" * 40)

    seed_everything(Config.SEED)

    # Define demo paths
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_meta = os.path.join(demo_dir, "train.parquet")
    demo_val_meta = os.path.join(demo_dir, "val.parquet")
    demo_test_meta = os.path.join(demo_dir, "test.parquet")

    # 2. Create Mini Datasets (Subset of real data)
    # We use the existing metadata in ./metadata to create small subsets
    print("\n[Demo] Preparing Mini Datasets...")
    create_demo_metadata(
        os.path.join("./metadata", "train.parquet"), demo_train_meta, n_samples=8
    )
    create_demo_metadata(
        os.path.join("./metadata", "val.parquet"), demo_val_meta, n_samples=4
    )
    create_demo_metadata(
        os.path.join("./metadata", "test.parquet"), demo_test_meta, n_samples=4
    )

    # 3. Monkey-patch Config for Demo
    # We modify the Config class attributes directly to affect the library modules
    print("\n[Demo] Configuring Runtime Parameters...")

    # Save original values to restore if needed (not strictly necessary for a one-shot script)
    orig_working_dir = Config.WORKING_DIR

    Config.WORKING_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    Config.TRAIN_META_PATH = demo_train_meta
    Config.VAL_META_PATH = demo_val_meta
    Config.TEST_META_PATH = demo_test_meta

    # Optimize for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Print modified config
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 4. Verify Model Logic
    verify_model_architecture()

    # 5. Run Training
    print("\n[Demo] Running Training Pipeline...")
    # We set load_cached_data=False to force the data_loader to process the new mini-parquets
    # instead of looking for the full cached arrays in the original working dir.
    best_auc = run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,
    )

    # Verify training output
    assert os.path.exists(
        Config.MODEL_PATH
    ), "Training failed to save the best model checkpoint."
    print(f"Training finished successfully. Best AUC: {best_auc}")

    # 6. Run Inference
    print("\n[Demo] Running Inference Pipeline...")
    generate_submission(load_cached_data=False, batch_size=Config.BATCH_SIZE)

    # Verify submission output
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Inference failed to save submission file."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(sub_df.head())

    # Check schema
    assert list(sub_df.columns) == [
        "BraTS21ID",
        "MGMT_value",
    ], "Submission columns are incorrect."
    assert len(sub_df) == 4, f"Expected 4 predictions, got {len(sub_df)}."

    # Check value range
    probs = sub_df["MGMT_value"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]."

    print("\n[Demo] Verification Successful!")

    # Cleanup (Optional)
    # shutil.rmtree(demo_dir)


if __name__ == "__main__":
    run_demo()
