import os
import shutil
import numpy as np
import pandas as pd
import torch
import random
import sys

# Import provided library modules
from library.config import Config
from library.utils import compute_kinematics, load_dataset, decode_predictions
from library.dataset import get_dataloaders, GestureDataset
from library.model import RGHC_MN
from library.trainer import Trainer
from library.inference import generate_submission, predict_sequence


def setup_demo_environment():
    """
    Sets up a temporary environment with mini-datasets and overrides Config.
    """
    print("=== Setting up Demo Environment ===")

    # Define paths
    base_dir = "./working/demo_env"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    metadata_dir = os.path.join(base_dir, "metadata")
    cache_dir = os.path.join(base_dir, "cache")
    submission_dir = os.path.join(base_dir, "submission")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # 1. Create Mini Metadata (Subsampling real metadata)
    # We read the original files and take the top 5 rows
    for fname in ["train.csv", "val.csv", "test.csv"]:
        src = os.path.join("./metadata", fname)
        dst = os.path.join(metadata_dir, fname)
        if os.path.exists(src):
            df = pd.read_csv(src)
            # Take a small subset (e.g., 5 samples)
            mini_df = df.head(5)
            mini_df.to_csv(dst, index=False)
            print(f"Created mini {fname} with {len(mini_df)} samples.")
        else:
            print(f"Warning: Source {src} not found. Creating empty dummy.")
            pd.DataFrame(
                columns=["sample_id", "rgb_path", "data_path", "audio_path", "labels"]
            ).to_csv(dst, index=False)

    # 2. Override Config
    print("Overriding Config parameters for demo...")
    Config.WORKING_DIR = base_dir
    Config.SUBMISSION_DIR = submission_dir

    Config.TRAIN_METADATA_PATH = os.path.join(metadata_dir, "train.csv")
    Config.VAL_METADATA_PATH = os.path.join(metadata_dir, "val.csv")
    Config.TEST_METADATA_PATH = os.path.join(metadata_dir, "test.csv")

    Config.TRAIN_CACHE_PATH = os.path.join(cache_dir, "dataset_train.npz")
    Config.VAL_CACHE_PATH = os.path.join(cache_dir, "dataset_val.npz")
    Config.TEST_CACHE_PATH = os.path.join(cache_dir, "dataset_test.npz")

    Config.BEST_MODEL_PATH = os.path.join(cache_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Speed up training for demo
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.STRIDE_TRAIN = 64  # Larger stride to reduce number of windows

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_utils_logic():
    print("\n=== Testing Utils Logic ===")

    # 1. Test compute_kinematics
    # Create a dummy skeleton sequence: T=10, J=20, C=3
    # Case: Constant position -> Velocity should be 0, Acc should be 0
    T, J, C = 10, 20, 3
    pos = np.ones((T, J, C), dtype=np.float32)

    kinematics = compute_kinematics(pos)
    # Expected shape: (T, J, 9) -> Pos(3) + Vel(3) + Acc(3)
    assert kinematics.shape == (
        T,
        J,
        9,
    ), f"Kinematics shape mismatch: {kinematics.shape}"

    # Check Velocity (indices 3:6)
    vel = kinematics[:, :, 3:6]
    # First frame of velocity is padded (diff with 0 or edge), subsequent should be 0
    # The implementation uses edge padding for the first frame calculation.
    # padded_pos = [p[0], p[0], p[1]...] -> diff[0] = p[0]-p[0] = 0.
    assert np.allclose(vel, 0), "Velocity should be zero for constant position."

    # Check Acceleration (indices 6:9)
    acc = kinematics[:, :, 6:9]
    assert np.allclose(acc, 0), "Acceleration should be zero for constant velocity."

    print("compute_kinematics logic verified.")

    # 2. Test decode_predictions
    # Create dummy probabilities (T=10, Classes=21)
    # Class 1 for 6 frames, Class 0 (Background) for 4 frames
    probs = np.zeros((10, 21), dtype=np.float32)
    probs[0:6, 1] = 1.0  # Class 1
    probs[6:10, 0] = 1.0  # Class 0

    # Config.MIN_GESTURE_DURATION is 5
    seq = decode_predictions(probs)
    # Should detect Class 1. Class 0 is background and ignored.
    assert seq == [1], f"Expected [1], got {seq}"

    # Test short duration filtering
    probs_short = np.zeros((10, 21), dtype=np.float32)
    probs_short[0:3, 2] = 1.0  # Class 2 for 3 frames (< 5)
    probs_short[3:10, 0] = 1.0
    seq_short = decode_predictions(probs_short)
    assert seq_short == [], f"Expected [], got {seq_short}"

    print("decode_predictions logic verified.")


def test_data_loading():
    print("\n=== Testing Data Loading ===")

    # Load the mini train dataset
    # This triggers processing raw files and caching
    data = load_dataset(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE_PATH, load_cached_data=False
    )

    assert "skeletons" in data
    assert "audio" in data
    assert "labels" in data
    assert len(data["skeletons"]) > 0

    # Check dimensions of the first sample
    skel = data["skeletons"][0]
    aud = data["audio"][0]
    lbl = data["labels"][0]

    print(f"Sample 0 shapes: Skel={skel.shape}, Aud={aud.shape}, Lbl={lbl.shape}")

    assert skel.ndim == 3 and skel.shape[1:] == (20, 3)
    assert aud.ndim == 2 and aud.shape[1] == 13
    assert lbl.ndim == 1
    assert skel.shape[0] == aud.shape[0] == lbl.shape[0]

    print("Data loading verified.")


def test_model_forward():
    print("\n=== Testing Model Architecture ===")

    # Get dataloaders
    train_loader, _, _ = get_dataloaders()

    # Fetch one batch
    features, labels, _, _ = next(iter(train_loader))

    # Expected Input Dim: 20*9 + 13 = 193
    # Expected Shape: (Batch, Time, 193)
    print(f"Batch Input Shape: {features.shape}")
    assert features.shape[2] == 193

    # Instantiate Model
    model = RGHC_MN()
    model.eval()

    # Forward Pass
    with torch.no_grad():
        outputs = model(features)

    # Check outputs
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    p3 = outputs["stage3"]
    # Shape: (Batch, Time, NumClasses=21)
    print(f"Model Output Shape (Stage 3): {p3.shape}")
    assert p3.shape == (features.shape[0], features.shape[1], 21)

    # Check probability sum
    sum_probs = p3.sum(dim=2)
    assert torch.allclose(
        sum_probs, torch.ones_like(sum_probs), atol=1e-5
    ), "Softmax output should sum to 1"

    print("Model architecture verified.")


def test_training_loop():
    print("\n=== Testing Training Loop ===")

    # Initialize Trainer
    trainer = Trainer(load_cached_data=True)

    # Run fit (Config.NUM_EPOCHS is set to 1)
    trainer.fit(epochs=Config.NUM_EPOCHS)

    # Check if model saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print(f"Training loop completed. Model saved at {Config.BEST_MODEL_PATH}")


def test_inference_pipeline():
    print("\n=== Testing Inference Pipeline ===")

    # Ensure model exists (from previous step)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Model file missing for inference test.")

    # Run submission generation
    generate_submission(load_cached_data=False)

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    print(f"Submission file contains {len(lines)} lines.")
    if len(lines) > 0:
        print(f"Sample line: {lines[0].strip()}")
        parts = lines[0].strip().split(",")
        # First part should be session ID (e.g., SampleXXXXX)
        assert parts[0].startswith("Sample") or parts[0].startswith(
            "Session"
        ), f"Invalid ID format: {parts[0]}"

    print("Inference pipeline verified.")


if __name__ == "__main__":
    try:
        setup_demo_environment()

        test_utils_logic()
        test_data_loading()
        test_model_forward()
        test_training_loop()
        test_inference_pipeline()

        print("\nAll demonstrations and verifications passed successfully!")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
