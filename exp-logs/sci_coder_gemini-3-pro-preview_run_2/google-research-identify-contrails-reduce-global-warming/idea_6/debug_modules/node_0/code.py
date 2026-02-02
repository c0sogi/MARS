import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import ContrailDataset, get_transforms
from library.model import StripPoolingResNet18UNet
from library.loss import HybridLoss
from library.train import run_training
from library.inference import generate_submission


def main():
    # ==========================================
    # 0. Setup & Configuration Override
    # ==========================================
    print("[1/6] Setting up environment and overriding configuration for demo...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config for fast demonstration
    Config.SEED = 42
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect outputs to a demo directory in working/
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed
    set_seed(Config.SEED)
    print("Configuration updated for fast execution.")

    # ==========================================
    # 1. Verify Utilities (RLE Encoding)
    # ==========================================
    print("\n[2/6] Verifying RLE Encoding utility...")

    # Create a simple 4x4 mask
    # 0 1 0 0
    # 0 1 0 0
    # 0 0 0 0
    # 0 0 0 0
    # Flattened (Column-major/Fortran): 0,0,0,0 (col1), 1,1,0,0 (col2), ...
    # Indices (1-based): pixel 5 and 6 are 1s.
    synthetic_mask = np.zeros((4, 4), dtype=np.uint8)
    synthetic_mask[0, 1] = 1
    synthetic_mask[1, 1] = 1

    encoded = rle_encode(synthetic_mask)
    expected = "5 2"  # Start at 5, length 2

    assert (
        encoded == expected
    ), f"RLE Encoding failed. Expected '{expected}', got '{encoded}'"
    print(f"RLE Encoding verified. Input mask with 2 pixels encoded as: '{encoded}'")

    # ==========================================
    # 2. Verify Dataset & Transforms
    # ==========================================
    print("\n[3/6] Verifying Dataset and Transforms...")

    # Initialize dataset in debug mode
    ds = ContrailDataset(
        split="train", transform=get_transforms("train", Config), debug=True
    )

    assert len(ds) > 0, "Dataset is empty."
    print(f"Dataset loaded with {len(ds)} samples (Debug Mode).")

    # Fetch one sample
    sample = ds[0]
    image = sample["image"]
    mask = sample["mask"]
    record_id = sample["record_id"]

    # Verify shapes
    # Image: (C=6, H=256, W=256)
    # Mask: (C=1, H=256, W=256)
    assert image.shape == (6, 256, 256), f"Incorrect image shape: {image.shape}"
    assert mask.shape == (1, 256, 256), f"Incorrect mask shape: {mask.shape}"
    assert isinstance(image, torch.Tensor), "Image is not a Tensor"
    assert isinstance(mask, torch.Tensor), "Mask is not a Tensor"

    print(f"Sample '{record_id}' loaded successfully.")
    print(f"Image Tensor Shape: {image.shape}, Type: {image.dtype}")
    print(f"Mask Tensor Shape: {mask.shape}, Type: {mask.dtype}")

    # ==========================================
    # 3. Verify Model & Loss
    # ==========================================
    print("\n[4/6] Verifying Model Architecture and Loss Function...")

    device = torch.device("cpu")  # Use CPU for simple logic check to save GPU init time
    model = StripPoolingResNet18UNet(in_channels=6, pretrained=False).to(device)
    criterion = HybridLoss()

    # Create dummy batch (B=2)
    dummy_input = torch.randn(2, 6, 256, 256).to(device)
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    # Forward pass
    output = model(dummy_input)

    # Verify output shape (B, 1, H, W)
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape mismatch: {output.shape}"

    # Compute loss
    loss = criterion(output, dummy_target)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"Model forward pass successful. Output shape: {output.shape}")
    print(f"Loss computation successful. Loss value: {loss.item():.4f}")

    # ==========================================
    # 4. Run Training Loop (Integration)
    # ==========================================
    print("\n[5/6] Running Training Loop (Integration Test)...")

    # We use the provided run_training function which uses the Config we modified
    try:
        run_training(debug=True)
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")

    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("Training loop completed successfully. Checkpoint saved.")

    # ==========================================
    # 5. Run Inference (Integration)
    # ==========================================
    print("\n[6/6] Running Inference and Submission Generation...")

    try:
        generate_submission(debug=True)
    except Exception as e:
        raise RuntimeError(f"Inference generation failed: {e}")

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    required_cols = ["record_id", "encoded_pixels"]
    for col in required_cols:
        assert col in df_sub.columns, f"Submission missing column: {col}"

    # Check if we have predictions (some might be empty strings represented as '-')
    # In debug mode with random weights, predictions might be noise, but format should hold.
    assert not df_sub.empty, "Submission dataframe is empty."

    print("Inference completed successfully. Submission format verified.")
    print("\nAll demonstrations passed successfully!")


if __name__ == "__main__":
    main()
