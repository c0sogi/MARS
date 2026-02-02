import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.modules import AHCHDN
from library.loss import AnchoredMCRMSELoss
from library.train import run_training, run_inference

if __name__ == "__main__":
    # 1. Setup Environment
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    seed_everything(42)

    # Define paths for this demonstration
    DEMO_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Override Config parameters for a quick demonstration
    print(f"Configuring demonstration parameters...")
    Config.IDEA_DIR = CACHE_DIR
    Config.SUBMISSION_DIR = SUBMISSION_DIR
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.PATIENCE = 2  # Short patience for early stopping

    # Ensure directories exist
    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Verify Model Architecture
    print("Verifying Model Architecture...")
    try:
        model = AHCHDN()
        # Create dummy input: (Batch=2, Length=107, Channels=18)
        # Channels: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18
        dummy_input = torch.randn(2, 107, 18)
        # Create dummy pair_map: (Batch=2, Length=107), filled with -1 (unpaired)
        dummy_pair_map = torch.full((2, 107), -1, dtype=torch.long)

        # Perform forward pass
        output = model(dummy_input, dummy_pair_map)

        # Check output shape: (Batch, Length, 5 Targets)
        expected_shape = (2, 107, 5)
        assert (
            output.shape == expected_shape
        ), f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
        print("  Model forward pass successful. Output shape verified.")
    except Exception as e:
        raise AssertionError(f"Model verification failed: {e}")

    # 3. Verify Loss Function
    print("Verifying Loss Function...")
    try:
        criterion = AnchoredMCRMSELoss()
        # Create identical prediction and target -> Loss should be 0.0
        pred_perfect = torch.randn(2, 107, 5)
        loss_zero = criterion(pred_perfect, pred_perfect)
        assert torch.isclose(
            loss_zero, torch.tensor(0.0)
        ), f"Loss should be 0.0 for perfect prediction, got {loss_zero.item()}"

        # Create prediction off by 1.0 -> MSE=1.0, RMSE=1.0, Mean=1.0
        pred_ones = torch.ones(2, 107, 5)
        target_zeros = torch.zeros(2, 107, 5)
        loss_one = criterion(pred_ones, target_zeros)
        assert torch.isclose(
            loss_one, torch.tensor(1.0)
        ), f"Loss should be 1.0 for unit error, got {loss_one.item()}"
        print("  Loss function logic verified.")
    except Exception as e:
        raise AssertionError(f"Loss verification failed: {e}")

    # 4. Run Training Pipeline (Debug Mode)
    print("\nStarting Training Pipeline (Debug Mode)...")
    # Debug mode uses a small subset of the data for speed
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=True)

    # Verify that the best model was saved
    model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Training failed to save model at {model_path}")
    print(f"  Training complete. Model saved to {model_path}")

    # 5. Run Inference Pipeline
    print("\nStarting Inference Pipeline...")
    # Runs inference on the full test set (240 samples) using the saved model
    run_inference(batch_size=Config.BATCH_SIZE)

    # Verify submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Inference failed to save submission at {submission_path}"
        )

    # Validate submission format
    df_sub = pd.read_csv(submission_path)

    # Check dimensions: 240 test samples * 107 positions = 25680 rows
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print(f"  Inference complete. Submission saved to {submission_path}")
    print("  Submission format verified.")

    print("\nDemo execution completed successfully.")
