import sys
import os
import torch
import pandas as pd
import numpy as np
import timm

# Add current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import Library Components
from library.config import Config, seed_everything
from library.utils import process_patient
from library.dataset import get_dataloader
from library.model import VFPNet
from library.train import run_training
from library.predict import generate_submission


def patch_timm_pretrained():
    """
    Monkey-patch timm.create_model to force pretrained=False.
    This avoids downloading weights during the demo, optimizing for speed and
    preventing network errors in restricted environments.
    """
    original_create_model = timm.create_model

    def mocked_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = mocked_create_model
    print(">>> Patched timm.create_model to force pretrained=False")


def main():
    print(">>> Starting BraTS21 Pipeline Demonstration...")

    # 1. Configuration Setup for Fast Demo
    # We override Config attributes to run a minimal version of the pipeline.
    Config.DEBUG = True
    Config.DEBUG_SIZE = 4  # Process only 4 patients
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Patch timm to avoid downloads
    patch_timm_pretrained()

    # 2. Verify Data Processing Logic (library.utils)
    print("\n>>> [1/5] Verifying Data Processing (process_patient)...")
    # Load metadata manually to get a sample row
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_META_PATH}")

    train_df = pd.read_parquet(Config.TRAIN_META_PATH)
    sample_row = train_df.iloc[0]

    print(f"Processing single patient: {sample_row['BraTS21ID']}")
    volume = process_patient(sample_row)

    # Assertions for 3D Volume Shape: (Channels, Depth, Height, Width)
    expected_shape = (4, Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE)
    assert isinstance(volume, torch.Tensor), "Output must be a torch.Tensor"
    assert (
        volume.shape == expected_shape
    ), f"Volume shape mismatch. Expected {expected_shape}, got {volume.shape}"
    print("✓ Data processing logic verified.")

    # 3. Verify Dataset and DataLoader (library.dataset)
    print("\n>>> [2/5] Verifying Dataset and DataLoader...")
    # Initialize DataLoader (triggers load_dataset with caching)
    # We use load_cached_data=False initially to force the processing logic to run
    train_loader = get_dataloader("train", shuffle=False, load_cached_data=False)

    # Fetch one batch
    batch_images, batch_labels = next(iter(train_loader))

    # Assertions for Batch Shapes
    # Images: (B, C, D, H, W)
    expected_batch_img_shape = (
        Config.BATCH_SIZE,
        4,
        Config.NUM_SLICES,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    # Labels: (B,)
    expected_batch_lbl_shape = (Config.BATCH_SIZE,)

    assert (
        batch_images.shape == expected_batch_img_shape
    ), f"Batch image shape mismatch. Expected {expected_batch_img_shape}, got {batch_images.shape}"
    assert (
        batch_labels.shape == expected_batch_lbl_shape
    ), f"Batch label shape mismatch. Expected {expected_batch_lbl_shape}, got {batch_labels.shape}"
    print("✓ DataLoader verified.")

    # 4. Verify Model Architecture (library.model)
    print("\n>>> [3/5] Verifying Model Architecture (VFPNet)...")
    model = VFPNet(num_classes=1, pretrained=False)
    model.eval()

    # Perform Forward Pass
    with torch.no_grad():
        outputs = model(batch_images)

    # Assertions for Output Shape: (B, 1)
    expected_output_shape = (Config.BATCH_SIZE, 1)
    assert (
        outputs.shape == expected_output_shape
    ), f"Model output shape mismatch. Expected {expected_output_shape}, got {outputs.shape}"
    print("✓ Model forward pass verified.")

    # 5. Verify Training Pipeline (library.train)
    print("\n>>> [4/5] Verifying Training Pipeline (run_training)...")
    # This runs the full training loop (1 epoch, debug subset)
    run_training()

    # Verify that the model checkpoint was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), f"Training failed: {best_model_path} was not created."
    print("✓ Training pipeline verified. Model saved.")

    # 6. Verify Inference Pipeline (library.predict)
    print("\n>>> [5/5] Verifying Inference Pipeline (generate_submission)...")
    # Generate submission using the model we just trained
    generate_submission(weights_path=best_model_path)

    # Verify submission file existence and format
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Inference failed: {submission_path} was not created."

    sub_df = pd.read_csv(submission_path)

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check row count (Should match DEBUG_SIZE because Config.DEBUG applies to test set loading too)
    assert (
        len(sub_df) == Config.DEBUG_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SIZE}, got {len(sub_df)}"

    print("✓ Inference pipeline verified. Submission generated.")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
