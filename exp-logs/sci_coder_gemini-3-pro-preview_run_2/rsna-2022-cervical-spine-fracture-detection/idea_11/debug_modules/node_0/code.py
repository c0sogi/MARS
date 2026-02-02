import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, weighted_log_loss, load_dicom
from library.dataset import CervicalSpineDataset
from library.model import CervicalFractureNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata():
    """
    Creates a small subset of the training and validation metadata.
    This allows us to run the training loop quickly for demonstration purposes.
    """
    print("Creating mini metadata for rapid testing...")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load original metadata
    train_full = pd.read_csv(Config.TRAIN_METADATA)
    val_full = pd.read_csv(Config.VAL_METADATA)

    # Create subsets (4 samples for train, 2 for val to satisfy batch_size=2)
    mini_train = train_full.head(4).copy()
    mini_val = val_full.head(2).copy()

    # Save mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)

    return mini_train_path, mini_val_path


def test_utils():
    """Verifies utility functions."""
    print("\n=== Testing Utils ===")

    # 1. Reproducibility Check
    seed_everything(42)
    val1 = np.random.rand()
    seed_everything(42)
    val2 = np.random.rand()
    assert val1 == val2, "seed_everything failed to ensure reproducibility."
    print("Reproducibility check passed.")

    # 2. Loss Function Logic Check
    # Scenario: Patient has overall fracture (index 7).
    y_true = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])

    # Good prediction (high prob for overall)
    y_pred_good = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99]])
    loss_good = weighted_log_loss(y_true, y_pred_good)

    # Bad prediction (low prob for overall)
    y_pred_bad = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
    loss_bad = weighted_log_loss(y_true, y_pred_bad)

    assert (
        loss_good < loss_bad
    ), "Weighted log loss logic failed: Good prediction had higher loss."
    print(f"Loss function check passed. (Good: {loss_good:.4f} < Bad: {loss_bad:.4f})")

    # 3. DICOM Loading Check
    # Use the first file from the training set
    df = pd.read_csv(Config.TRAIN_METADATA)
    if not df.empty:
        rel_path = df.iloc[0]["image_path"]
        full_dir = os.path.join(Config.DATA_ROOT, rel_path)
        if os.path.exists(full_dir):
            files = [f for f in os.listdir(full_dir) if f.endswith(".dcm")]
            if files:
                img_path = os.path.join(full_dir, files[0])
                # Load and resize to small size
                img = load_dicom(img_path, size=(128, 128))

                assert isinstance(
                    img, np.ndarray
                ), "load_dicom returned incorrect type."
                assert img.shape == (
                    128,
                    128,
                ), f"load_dicom returned shape {img.shape}, expected (128, 128)."
                assert (
                    0.0 <= img.min() and img.max() <= 1.0
                ), "DICOM image not normalized to [0, 1]."
                print("DICOM loading check passed.")


def test_dataset(mini_train_path):
    """Verifies Dataset loading and transforms."""
    print("\n=== Testing Dataset ===")

    # Initialize dataset with mini metadata
    # Config.DEBUG is True, so SEQ_LEN=32, IMAGE_SIZE=(256, 256)
    ds = CervicalSpineDataset(mini_train_path, mode="train", load_cached_data=False)

    assert len(ds) == 4, f"Dataset length mismatch. Expected 4, got {len(ds)}."

    # Fetch first item
    volume, labels = ds[0]

    print(f"Sample Volume Shape: {volume.shape}")
    print(f"Sample Labels: {labels}")

    # Validate Shapes
    expected_seq = Config.SEQ_LEN  # 32 in debug
    expected_h, expected_w = Config.IMAGE_SIZE  # (256, 256) in debug

    assert volume.shape == (
        expected_seq,
        3,
        expected_h,
        expected_w,
    ), f"Volume shape mismatch. Got {volume.shape}"
    assert labels.shape == (8,), f"Labels shape mismatch. Got {labels.shape}"

    print("Dataset verification passed.")
    return volume.unsqueeze(0)  # Return batch for model test


def test_model(sample_batch):
    """Verifies Model architecture and forward pass."""
    print("\n=== Testing Model ===")

    model = CervicalFractureNet()
    model.eval()

    # Use CPU for this quick check to avoid CUDA overhead if not needed
    # (though Config defaults to CUDA if available)
    device = torch.device("cpu")
    model.to(device)
    sample_batch = sample_batch.to(device)

    with torch.no_grad():
        output = model(sample_batch)

    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (
        1,
        8,
    ), f"Model output shape mismatch. Expected (1, 8), got {output.shape}"

    # Check Sigmoid range
    assert torch.all(output >= 0) and torch.all(
        output <= 1
    ), "Model outputs are not valid probabilities [0, 1]."

    print("Model verification passed.")


def test_training_pipeline(mini_train_path, mini_val_path):
    """Verifies the Trainer loop."""
    print("\n=== Testing Training Pipeline ===")

    # Inject mini metadata paths into Config
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path

    # Initialize Trainer
    trainer = Trainer()

    # Run training
    # Config.EPOCHS is set to 1 in main
    # Dataset size is 4, Batch size is 2 -> 2 steps per epoch
    trainer.fit()

    # Verify artifacts
    if os.path.exists(Config.MODEL_PATH):
        print(f"Training successful. Model saved to: {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Training completed but model file was not found.")


if __name__ == "__main__":
    # 1. Setup Configuration for Speed
    # Enable debug mode: Reduces SEQ_LEN to 32, IMAGE_SIZE to 256, EPOCHS to 2
    Config.setup(debug=True)

    # Further optimizations for the demo
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # 2. Prepare Data
    mini_train_path, mini_val_path = create_mini_metadata()

    # 3. Execute Tests
    test_utils()
    sample_batch = test_dataset(mini_train_path)
    test_model(sample_batch)
    test_training_pipeline(mini_train_path, mini_val_path)
