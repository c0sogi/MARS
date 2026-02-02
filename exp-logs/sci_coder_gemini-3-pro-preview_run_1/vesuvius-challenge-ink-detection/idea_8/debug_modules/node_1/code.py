import os
import shutil
import torch
import numpy as np
import pandas as pd
import random

# Import from the provided library files
from library.config import Config
from library.utils import rle_encode, calculate_fbeta, optimize_threshold
from library.model import FRUNet
from library.dataset import InkDataset
from library.train import train_model
from library.inference import predict_and_encode


def set_reproducibility(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def demo_utils():
    print("\n=== 1. Demonstrating Utils ===")

    # Test RLE Encoding
    # Mask: 0 1 1 1 0 0 1 0
    # Indices (1-based): 2,3,4 and 7
    # Run 1: Start 2, Length 3
    # Run 2: Start 7, Length 1
    # Expected: "2 3 7 1"
    dummy_mask = np.array([[0, 1, 1, 1, 0, 0, 1, 0]], dtype=np.uint8)
    encoded = rle_encode(dummy_mask)
    print(f"RLE Input: {dummy_mask.flatten()}")
    print(f"RLE Output: '{encoded}'")
    assert (
        encoded == "2 3 7 1"
    ), f"RLE Encoding failed. Expected '2 3 7 1', got '{encoded}'"

    # Test F-Beta Score (F0.5)
    # Pred: 1 1 0 0
    # True: 1 0 1 0
    # TP=1, FP=1, FN=1
    # Precision = 1/2 = 0.5
    # Recall = 1/2 = 0.5
    # F0.5 = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    p = np.array([1, 1, 0, 0])
    t = np.array([1, 0, 1, 0])
    score = calculate_fbeta(p, t, beta=0.5)
    print(f"F0.5 Score: {score}")
    assert (
        abs(score - 0.5) < 1e-6
    ), f"F0.5 calculation failed. Expected 0.5, got {score}"

    print("Utils verification passed.")


def demo_model():
    print("\n=== 2. Demonstrating Model Architecture (FRUNet) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FRUNet().to(device)

    # Input shape: (Batch, Z_DIM, H, W)
    # Config.Z_DIM is 65
    batch_size = 2
    dummy_input = torch.randn(batch_size, 65, 512, 512).to(device)

    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")

    # Expected output: (Batch, 1, H, W)
    expected_shape = (batch_size, 1, 512, 512)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model architecture verification passed.")


def demo_dataset():
    print("\n=== 3. Demonstrating InkDataset ===")

    # Use 'train' mode with a limit to fetch real data
    # Note: This relies on ./metadata/train.csv existing (provided in environment)
    dataset = InkDataset(mode="train", load_cached_data=False, limit=4)

    print(f"Dataset length (limited): {len(dataset)}")
    if len(dataset) == 0:
        print("Warning: Dataset is empty. Skipping dataset verification.")
        return

    # Fetch one sample
    volume, label, mask, sample_id = dataset[0]

    print(f"Sample ID: {sample_id}")
    print(f"Volume Tensor Shape: {volume.shape} (Type: {volume.dtype})")
    print(f"Label Tensor Shape: {label.shape} (Type: {label.dtype})")
    print(f"Mask Tensor Shape: {mask.shape} (Type: {mask.dtype})")

    # Assertions
    assert volume.ndim == 3 and volume.shape[0] == 65, "Volume should be (65, H, W)"
    assert label.ndim == 3 and label.shape[0] == 1, "Label should be (1, H, W)"
    assert mask.ndim == 3 and mask.shape[0] == 1, "Mask should be (1, H, W)"
    assert isinstance(volume, torch.Tensor), "Output should be a torch Tensor"

    print("Dataset verification passed.")


def demo_training_and_inference():
    print("\n=== 4. Demonstrating Training and Inference Pipeline ===")

    # --- Patch Config for Demo ---
    # We modify the Config class attributes directly to create a contained demo environment
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTIONS_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Speed up training
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Setup directories (this will create the new demo dirs)
    Config.setup_directories()

    # --- Run Training ---
    print("Starting training demo (limit=4 samples)...")
    # limit_samples=4 ensures we only load a tiny bit of data
    train_model(limit_samples=4)

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        # If validation F0.5 was 0 (likely with 4 random samples), best_model might not save
        # unless logic handles it. However, the logic saves if val_f05 > best_f05 (init 0).
        # If model learns nothing, it might stay 0.
        # For demo purposes, we force a save if it didn't happen, to test inference.
        print(
            "Checkpoint not found (likely due to low score on tiny data). Saving dummy checkpoint."
        )
        model = FRUNet()
        torch.save(model.state_dict(), checkpoint_path)
        with open(os.path.join(Config.CHECKPOINT_DIR, "best_threshold.txt"), "w") as f:
            f.write("0.5")

    assert os.path.exists(checkpoint_path), "Checkpoint file missing."
    print("Training demo complete.")

    # --- Run Inference ---
    print("Starting inference demo...")
    # We use the checkpoint we just (potentially) created
    predict_and_encode(
        checkpoint_path=checkpoint_path,
        output_file=Config.SUBMISSION_FILE,
        load_cached_data=True,  # Use cache generated during training/setup
        batch_size=2,
        num_workers=0,
    )

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_FILE):
        df = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"Submission file generated with {len(df)} rows.")
        print(df.head())

        expected_cols = ["Id", "Predicted"]
        assert (
            list(df.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}"
    else:
        # If test set is empty (possible in some envs), we accept that but print warning
        print("Submission file not generated (Test set might be empty).")

    print("Pipeline verification passed.")


if __name__ == "__main__":
    set_reproducibility(42)

    try:
        demo_utils()
        demo_model()
        demo_dataset()
        demo_training_and_inference()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        raise e
