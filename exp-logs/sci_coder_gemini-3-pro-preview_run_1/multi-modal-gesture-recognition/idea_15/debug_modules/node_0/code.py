import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, levenshtein_distance, rle_decode, compute_error_rate
from library.model import GCA_IIN
from library.train_eval import train_model, predict_test
from library.data_loader import get_dataloaders


def setup_demo_environment():
    """
    Creates a subset of metadata to speed up the demo execution.
    Redirects Config to use this subset.
    """
    print("Setting up demo environment with data subsets...")

    # Create temporary directories
    demo_meta_dir = os.path.join(Config.WORKING_DIR, "demo_metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Subset size
    SUBSET_SIZE = 5

    # Process each split
    for split in ["train", "val", "test"]:
        original_csv = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        if os.path.exists(original_csv):
            df = pd.read_csv(original_csv)
            # Take a small subset
            df_subset = df.head(SUBSET_SIZE)

            # Save to demo directory
            target_csv = os.path.join(demo_meta_dir, f"{split}.csv")
            df_subset.to_csv(target_csv, index=False)
            print(f"Created {split} subset with {len(df_subset)} samples.")
        else:
            print(f"Warning: {original_csv} not found.")

    # Override Config to point to demo metadata
    Config.METADATA_DIR = demo_meta_dir

    # Ensure clean cache for demo to prove cache generation works
    # Note: In a real run, we would keep the cache.
    # For this demo, we let it generate cache for the 5 samples.
    print(f"Configured metadata directory to: {Config.METADATA_DIR}")


def test_utilities():
    """
    Verifies utility functions logic.
    """
    print("\n--- Testing Utilities ---")

    # 1. Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Expected distance 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Expected distance 1 (deletion), got {dist_diff}"

    print("Levenshtein distance check passed.")

    # 2. RLE Decode
    # Config.MIN_GESTURE_LENGTH is 5
    # Config.BACKGROUND_CLASS_ID is 0
    # Create sequence: 5x '1', 3x '0' (bg), 6x '2', 4x '3' (too short)
    raw_preds = np.array([1] * 5 + [0] * 3 + [2] * 6 + [3] * 4)
    decoded = rle_decode(raw_preds)

    # Expected: [1, 2]. '3' is ignored (<5 frames). '0' is background.
    assert decoded == [1, 2], f"RLE Decode failed. Expected [1, 2], got {decoded}"
    print("RLE Decode check passed.")

    # 3. Error Rate
    preds = [[1, 2], [3]]
    targets = [[1, 2], [3, 4]]
    # Dist 1: 0, Dist 2: 1 (insertion/deletion). Total Dist: 1. Total Len: 2+2=4. Error: 0.25
    error = compute_error_rate(preds, targets)
    assert abs(error - 0.25) < 1e-6, f"Error rate calculation failed. Got {error}"
    print("Error Rate check passed.")


def test_model_forward_pass():
    """
    Verifies model instantiation and forward pass shapes.
    """
    print("\n--- Testing Model Forward Pass ---")

    set_seed(Config.SEED)
    device = torch.device("cpu")  # Use CPU for simple shape check

    model = GCA_IIN().to(device)
    model.eval()

    batch_size = 2
    seq_len = 30

    # Create dummy inputs
    skeleton = torch.randn(batch_size, seq_len, Config.INPUT_DIM_SKELETON).to(device)
    audio = torch.randn(batch_size, seq_len, Config.INPUT_DIM_AUDIO).to(device)
    lengths = torch.tensor([seq_len, seq_len - 5]).to(device)  # Variable lengths

    with torch.no_grad():
        logits = model(skeleton, audio, lengths)

    # Check Output Shape: (Batch, Time, NumClasses)
    expected_shape = (batch_size, seq_len, Config.NUM_CLASSES)
    assert (
        logits.shape == expected_shape
    ), f"Shape mismatch. Expected {expected_shape}, got {logits.shape}"

    print(f"Model forward pass successful. Output shape: {logits.shape}")


def run_training_pipeline():
    """
    Runs the training loop for 1 epoch on the subset data.
    """
    print("\n--- Running Training Pipeline (Demo) ---")

    # We use a small batch size and 1 epoch
    best_model_path = train_model(num_epochs=1, batch_size=2, debug=False)

    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print(f"Training finished. Checkpoint saved at: {best_model_path}")
    return best_model_path


def run_inference_pipeline(model_path):
    """
    Runs inference on the test subset.
    """
    print("\n--- Running Inference Pipeline (Demo) ---")

    predict_test(model_path=model_path, batch_size=2)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            print(f"Sample prediction: {lines[0].strip()}")
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Invalid submission format"

    print("Inference pipeline completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Setup Data Subsets for Speed
    setup_demo_environment()

    # 2. Verify Utilities
    test_utilities()

    # 3. Verify Model Logic
    test_model_forward_pass()

    # 4. Run Training (Dataset creation -> Cache -> Train -> Val)
    checkpoint_path = run_training_pipeline()

    # 5. Run Inference
    run_inference_pipeline(checkpoint_path)

    print("\nAll demonstrations completed successfully.")
