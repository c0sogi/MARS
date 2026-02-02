import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data_loader import GestureDataset, collate_fn
from library.model import GHG_CRCN
from library.loss import HierarchicalLoss
from library.inference import run_pipeline


def main():
    print("=== GHG-CRCN Pipeline Demonstration ===")

    # 1. Configuration Overrides for Speed and Demonstration
    # We modify the global Config to run on a tiny subset with a smaller model
    print(">>> Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.LSTM_HIDDEN_SIZE = 32  # Reduce model complexity
    Config.TCN_CHANNELS = 32
    Config.TCN_LAYERS = 2

    # Clean working directory to ensure fresh cache generation
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading Verification
    print("\n>>> Testing Data Loader...")
    # Force processing from scratch to verify parsing logic
    train_dataset = GestureDataset(split="train", debug=True, load_cached_data=False)

    assert len(train_dataset) > 0, "Dataset should not be empty."
    print(f"Loaded {len(train_dataset)} samples.")

    # Inspect a single sample
    sample = train_dataset[0]
    features = sample["features"]  # (Time, InputDim)
    targets_cls = sample["targets_cls"]  # (Time,)

    print(f"Sample Features Shape: {features.shape}")
    print(f"Sample Targets Shape: {targets_cls.shape}")

    # Assertions
    assert (
        features.shape[1] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {features.shape[1]}"
    assert (
        features.shape[0] == targets_cls.shape[0]
    ), "Feature and target time dimensions mismatch."

    # Test Collate Function
    batch_list = [train_dataset[0], train_dataset[1]]
    batch = collate_fn(batch_list)

    b_features = batch["features"]
    b_mask = batch["mask"]

    print(f"Batch Features Shape: {b_features.shape}")
    print(f"Batch Mask Shape: {b_mask.shape}")

    assert b_features.dim() == 3, "Batch features should be 3D (Batch, Time, Dim)"
    assert b_mask.dim() == 2, "Batch mask should be 2D (Batch, Time)"
    assert b_features.size(0) == 2, "Batch size mismatch."

    # 3. Model Forward Pass Verification
    print("\n>>> Testing Model Forward Pass...")
    model = GHG_CRCN().to(device)

    # Move inputs to device
    b_features = b_features.to(device)
    b_mask = b_mask.to(device)

    # Forward pass
    out1, out2, out3 = model(b_features, b_mask)

    print(f"Stage 1 Output: {out1.shape}")
    print(f"Stage 2 Output: {out2.shape}")
    print(f"Stage 3 Output: {out3.shape}")

    # Expected output channels: NumClasses (21) + Boundary (1) + Foreground (1) = 23
    expected_channels = Config.NUM_CLASSES + 2
    assert (
        out3.shape[2] == expected_channels
    ), f"Expected output channels {expected_channels}, got {out3.shape[2]}"
    assert out3.shape[1] == b_features.shape[1], "Output time dimension mismatch."

    # 4. Loss Computation Verification
    print("\n>>> Testing Loss Computation...")
    criterion = HierarchicalLoss().to(device)

    b_targets_cls = batch["targets_cls"].to(device)
    b_targets_bnd = batch["targets_bnd"].to(device)
    b_targets_fg = batch["targets_fg"].to(device)

    loss = criterion(
        [out1, out2, out3], b_targets_cls, b_targets_bnd, b_targets_fg, b_mask
    )

    print(f"Computed Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss must be positive."

    # Verify Backward Pass
    loss.backward()
    print("Backward pass successful.")

    # 5. Full Pipeline Execution
    print("\n>>> Running Full Pipeline (Train + Inference)...")
    # This runs Trainer.fit() and Trainer.predict()
    try:
        run_pipeline(epochs=Config.NUM_EPOCHS, patience=1, debug=True)

        # Verify Outputs
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

        if os.path.exists(checkpoint_path):
            print(f"Checkpoint verified at: {checkpoint_path}")
        else:
            print(
                "Notice: No checkpoint saved (Validation score might not have improved in 1 epoch)."
            )

        if os.path.exists(submission_path):
            print(f"Submission file verified at: {submission_path}")

            # Check content format
            with open(submission_path, "r") as f:
                lines = f.readlines()
                print(f"Submission contains {len(lines)} lines.")
                if len(lines) > 0:
                    print(f"First line: {lines[0].strip()}")
                    assert (
                        "," in lines[0]
                    ), "Submission format incorrect (missing comma)."
        else:
            raise FileNotFoundError("Submission file was not generated.")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
