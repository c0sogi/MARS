import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Ensure local library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import CervicalSpineDataset
from library.model import CervicalSpineTransformer
from library.loss import WeightedMultiLabelLogLoss
import library.engine as engine


def demo_pipeline():
    print("=== Cervical Spine Fracture Detection Demo ===\n")

    # 1. Configuration for Demo Speed
    # We modify the global Config class attributes to run a lightweight version of the task
    print("1. Configuring environment for fast demonstration...")
    Config.DEBUG = True  # Use a tiny subset of data (20 samples)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.SEQ_LEN = 8  # Reduced sequence length (from 24)
    Config.IMAGE_SIZE = 256  # Reduced resolution (from 384)
    Config.NUM_WORKERS = 0  # Disable multiprocessing to reduce overhead

    # Redirect output to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "output")
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Manually create these directories since Config class logic ran at import time
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("   Configuration complete. Output will be saved to:", Config.WORKING_DIR)

    # 2. Dataset Verification
    print("\n2. Verifying Dataset Loading...")
    # Initialize dataset in debug mode
    train_dataset = CervicalSpineDataset(split="train", debug=True)
    print(f"   Train dataset size: {len(train_dataset)}")

    # Fetch one sample to check shapes
    sample_img, sample_target = train_dataset[0]

    # Expected: (Seq_Len, Channels, H, W) and (8 targets)
    expected_img_shape = (Config.SEQ_LEN, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    expected_target_shape = (8,)

    print(f"   Sample Image Shape: {sample_img.shape}")
    print(f"   Sample Target Shape: {sample_target.shape}")

    assert (
        sample_img.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {sample_img.shape}"
    assert (
        sample_target.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {sample_target.shape}"
    print("   Dataset verification passed.")

    # 3. Model Architecture Verification
    print("\n3. Verifying Model Architecture...")
    device = Config.DEVICE
    model = CervicalSpineTransformer().to(device)

    # Create a dummy batch of size 1
    dummy_input = sample_img.unsqueeze(0).to(device)  # (1, Seq, C, H, W)

    print("   Forward pass with dummy input...")
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"   Model Output Shape: {logits.shape}")
    # Expected output: (Batch_Size, Num_Queries) -> (1, 8)
    assert logits.shape == (
        1,
        8,
    ), f"Model output shape mismatch. Expected (1, 8), got {logits.shape}"
    print("   Model verification passed.")

    # 4. Loss Function Verification
    print("\n4. Verifying Loss Function...")
    criterion = WeightedMultiLabelLogLoss().to(device)
    dummy_target = sample_target.unsqueeze(0).to(device)

    # Calculate loss
    loss = criterion(logits, dummy_target)
    print(f"   Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print("   Loss verification passed.")

    # 5. Full Pipeline Execution (Engine)
    print("\n5. Running Full Training/Inference Pipeline...")
    print("   (This runs training on the debug subset, validation, and test inference)")

    # engine.run handles the loop, checkpointing, and submission generation
    # We pass the modified Config parameters implicitly via the class attributes
    engine.run(epochs=Config.EPOCHS, debug=Config.DEBUG)

    # 6. Submission Verification
    print("\n6. Verifying Submission Output...")
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   Submission file found with {len(sub_df)} rows.")

        expected_cols = ["row_id", "fractured"]
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

        # Check if probabilities are valid
        if not sub_df.empty:
            assert (
                sub_df["fractured"].min() >= 0 and sub_df["fractured"].max() <= 1
            ), "Probabilities out of range [0, 1]"
            print("   First 5 rows:")
            print(sub_df.head())

        print("   Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n=== All demonstrations completed successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        demo_pipeline()
    except Exception as e:
        print(f"\nERROR: Demonstration failed with exception: {e}")
        raise e
