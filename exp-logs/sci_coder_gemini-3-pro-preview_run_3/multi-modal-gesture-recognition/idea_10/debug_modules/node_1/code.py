import os
import sys
import shutil
import torch
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import (
    align_to_canonical_view,
    compute_kinematics,
    rle_decode,
    TruncatedMSELoss,
)
from library.dataset import GestureDataset
from library.model import VIARN
from library.trainer import Trainer


def main():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    print("Setting up configuration for demo...")

    # Override Config for fast execution
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 5  # Use only 5 samples for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.WORKING_DIR = "./working/demo_run_script"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set fixed seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # ==========================================
    # 2. Verify Utils
    # ==========================================
    print("Verifying utility functions...")

    # Mock Skeleton Data: (Time=10, Joints=20, Coords=3)
    mock_skeleton = np.random.rand(10, 20, 3).astype(np.float32)

    # Test Alignment
    aligned_skel = align_to_canonical_view(mock_skeleton)
    assert aligned_skel.shape == (10, 20, 3), "Alignment output shape mismatch"

    # Test Kinematics
    # Output should be (Time, Joints, 9) -> [Pos, Vel, Acc]
    kinematics = compute_kinematics(mock_skeleton)
    assert kinematics.shape == (10, 20, 9), "Kinematics output shape mismatch"

    # Test RLE Decode
    # 0 is background, should be removed. Collapses consecutive duplicates.
    raw_preds = [0, 0, 1, 1, 2, 0, 3, 3, 0]
    decoded = rle_decode(raw_preds)
    assert decoded == [1, 2, 3], f"RLE Decode failed. Got {decoded}"

    print("Utils verified.")

    # ==========================================
    # 3. Verify Model
    # ==========================================
    print("Verifying Model architecture...")

    model = VIARN()
    # Input: (Batch, InputDim, Time)
    # InputDim = 193 (180 skeleton + 13 audio)
    dummy_input = torch.randn(2, Config.INPUT_DIM, 64)

    out1, out2, out3 = model(dummy_input)

    # Output: (Batch, NumClasses, Time)
    expected_shape = (2, Config.NUM_CLASSES, 64)
    assert out1.shape == expected_shape, f"Stage 1 output mismatch: {out1.shape}"
    assert out2.shape == expected_shape, f"Stage 2 output mismatch: {out2.shape}"
    assert out3.shape == expected_shape, f"Stage 3 output mismatch: {out3.shape}"

    print("Model verified.")

    # ==========================================
    # 4. Patch Dataset & Run Trainer
    # ==========================================
    print("Initializing Trainer and Dataset...")

    # Monkey Patch GestureDataset.__getitem__
    # The provided dataset returns (Time, Channels), but the Model expects (Channels, Time).
    # The Trainer passes data directly from loader to model without transposition.
    # We patch __getitem__ to perform the transposition (Permute Time and Channels).

    original_getitem = GestureDataset.__getitem__

    def patched_getitem(self, idx):
        features, labels = original_getitem(self, idx)
        # features is (Time, Channels) -> Permute to (Channels, Time)
        features = features.permute(1, 0)
        return features, labels

    GestureDataset.__getitem__ = patched_getitem

    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    print("Running training (1 epoch)...")
    trainer.train(num_epochs=1)

    # Ensure model is saved for prediction (in case validation didn't trigger save)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)

    # Run Inference
    print("Running inference on test set...")
    trainer.predict_test()

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Success! Submission generated at: {Config.SUBMISSION_PATH}")
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            print(f"Submission has {len(lines)} lines.")
            if len(lines) > 0:
                print(f"Sample line: {lines[0].strip()}")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
