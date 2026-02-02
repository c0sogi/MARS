import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, dice_score_batch
from library.dataset import ContrailDataset
from library.model import SoftGatedResnetUNet
from library.loss import SoftGatedLoss
from library.train import train_model
from library.predict import predict


def setup_demo_environment():
    """
    Sets up temporary directories and overrides Config for a fast demo run.
    """
    print(">>> Setting up demo environment...")

    # Define demo directories
    demo_working_dir = "./working/demo_run"
    demo_submission_dir = "./working/demo_submission"

    # Clean up previous runs if they exist
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    if os.path.exists(demo_submission_dir):
        shutil.rmtree(demo_submission_dir)

    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_FILE = os.path.join(demo_submission_dir, "submission.csv")

    # Override Config parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce workers for small demo

    # Create a subset of test metadata for fast inference demo
    # We read the original test metadata, take top 10 rows, and save to a new file
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    demo_test_df = full_test_df.head(10).copy()
    demo_test_meta_path = os.path.join(demo_working_dir, "demo_test_metadata.csv")
    demo_test_df.to_csv(demo_test_meta_path, index=False)

    # Point Config to this new metadata file
    Config.TEST_METADATA_PATH = demo_test_meta_path

    print(f"    Working Dir: {Config.WORKING_DIR}")
    print(
        f"    Test Metadata Subset: {Config.TEST_METADATA_PATH} ({len(demo_test_df)} samples)"
    )
    print("-" * 40)


def test_rle_encoding():
    """
    Verifies the RLE encoding logic with a simple manual example.
    """
    print(">>> Testing RLE Encoding...")

    # Create a 4x4 mask
    # Column-major flattening (Fortran):
    # Col 0: 0,1,0,0
    # Col 1: 0,1,0,0
    # Col 2: 0,0,0,0
    # Col 3: 0,0,0,0
    # Flattened: 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # Indices (1-based):
    # 2 is 1 (start 2, len 1)
    # 6 is 1 (start 6, len 1)
    mask = np.array(
        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )

    encoded = rle_encode(mask)
    expected = "2 1 6 1"

    assert (
        encoded == expected
    ), f"RLE Encoding failed. Expected '{expected}', got '{encoded}'"
    print("    RLE Encoding verified successfully.")
    print("-" * 40)


def test_dataset_and_dataloader():
    """
    Verifies Dataset loading and DataLoader batching.
    """
    print(">>> Testing Dataset and DataLoader...")

    # Initialize dataset with a small sample limit
    ds = ContrailDataset(split="train", max_samples=10)
    assert len(ds) == 10, f"Dataset length mismatch. Expected 10, got {len(ds)}"

    # Check single item
    img, mask, rid = ds[0]

    # Check shapes
    # Image: (C, H, W) -> (6, 256, 256)
    # Mask: (C, H, W) -> (1, 256, 256)
    assert img.shape == (6, 256, 256), f"Image shape incorrect: {img.shape}"
    assert mask.shape == (1, 256, 256), f"Mask shape incorrect: {mask.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    assert isinstance(mask, torch.Tensor), "Mask is not a Tensor"

    print(f"    Sample loaded. Image: {img.shape}, Mask: {mask.shape}, ID: {rid}")
    print("    Dataset verified successfully.")
    print("-" * 40)


def test_model_architecture():
    """
    Verifies Model instantiation, forward pass, and loss calculation.
    """
    print(">>> Testing Model Architecture and Loss...")

    device = Config.DEVICE
    model = SoftGatedResnetUNet(in_channels=6, pretrained=False).to(device)
    criterion = SoftGatedLoss()

    # Create dummy batch
    batch_size = 2
    dummy_img = torch.randn(batch_size, 6, 256, 256).to(device)
    dummy_mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)

    # Forward pass
    outputs = model(dummy_img)

    # Check outputs
    assert "mask" in outputs and "cls" in outputs, "Model output missing keys"
    pred_mask = outputs["mask"]
    pred_cls = outputs["cls"]

    assert pred_mask.shape == (
        batch_size,
        1,
        256,
        256,
    ), f"Pred mask shape wrong: {pred_mask.shape}"
    assert pred_cls.shape == (batch_size, 1), f"Pred cls shape wrong: {pred_cls.shape}"

    # Calculate loss
    loss = criterion(outputs, dummy_mask)

    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"    Forward pass successful. Loss: {loss.item():.4f}")
    print("    Model and Loss verified successfully.")
    print("-" * 40)


def run_training_demo():
    """
    Runs the training pipeline using the library function.
    """
    print(">>> Running Training Demo...")

    # Run training for 1 epoch on 16 samples
    # This will save 'best_model.pth' to Config.WORKING_DIR
    train_model(epochs=1, batch_size=4, max_samples=16, learning_rate=1e-4, patience=1)

    # Verify model file creation
    expected_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(expected_model_path), "best_model.pth was not created!"

    print("    Training demo completed successfully.")
    print("-" * 40)


def run_inference_demo():
    """
    Runs the inference pipeline using the library function.
    """
    print(">>> Running Inference Demo...")

    # Run prediction
    # This uses the 'best_model.pth' created in the previous step
    # and the subset metadata created in setup_demo_environment
    predict(batch_size=4)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found!"

    df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    expected_cols = ["record_id", "encoded_pixels"]
    assert all(
        col in df.columns for col in expected_cols
    ), "Submission columns mismatch"

    # Check row count (should match our demo metadata subset size of 10)
    assert len(df) == 10, f"Submission length mismatch. Expected 10, got {len(df)}"

    print("    Submission file content preview:")
    print(df.head(3))
    print("    Inference demo completed successfully.")
    print("-" * 40)


if __name__ == "__main__":
    # Ensure reproducibility
    Config.set_seed(42)

    # 1. Setup
    setup_demo_environment()

    # 2. Unit Tests
    test_rle_encoding()
    test_dataset_and_dataloader()
    test_model_architecture()

    # 3. Integration Tests (Train & Predict)
    run_training_demo()
    run_inference_demo()

    print("\n>>> All demonstrations passed successfully!")
