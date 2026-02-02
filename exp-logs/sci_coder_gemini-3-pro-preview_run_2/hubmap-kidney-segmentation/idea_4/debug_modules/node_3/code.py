import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_encode, dice_coef
from library.model import MultiTaskResNetFPN
from library.loss import MultiTaskLoss
from library.engine import Trainer
from library.inference import InferenceRunner


def run_demo():
    print("Starting HuBMAP Pipeline Demo...")

    # ====================================================
    # 1. Configuration Override
    # ====================================================
    print("\n[1] Configuring environment for demo run...")

    # Set a specific working directory for this demo
    DEMO_WORKDIR = "./working/demo_run"
    if os.path.exists(DEMO_WORKDIR):
        shutil.rmtree(DEMO_WORKDIR)
    os.makedirs(DEMO_WORKDIR, exist_ok=True)

    # Override Config parameters for speed and debugging
    Config.WORKING_DIR = DEMO_WORKDIR
    Config.MODEL_PATH = os.path.join(DEMO_WORKDIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORKDIR, "submission", "submission.csv")

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2  # Use only 2 images for training/val
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.WARMUP_EPOCHS = 0  # Disable warmup for demo to ensure model saving
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.TILE_SIZE = 512  # Smaller tile size for faster processing
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # ====================================================
    # 2. Component Verification (Unit Tests)
    # ====================================================
    print("\n[2] Verifying core components...")

    # --- Test Utils: RLE Encoding ---
    # Create a simple 4x4 mask:
    # 0 0 0 0
    # 1 1 0 0  -> flattened (col-major): 0,1,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0
    # 0 0 0 0
    # 0 0 0 0
    # Wait, RLE is column-major (Fortran).
    # Col 0: 0, 1, 0, 0
    # Col 1: 0, 1, 0, 0
    # Col 2: 0, 0, 0, 0
    # Col 3: 0, 0, 0, 0
    # Flat: 0, 1, 0, 0, 0, 1, 0, 0, ...
    # Run 1: Start 2, Len 1. Run 2: Start 6, Len 1.
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[1, 0] = 1
    dummy_mask[1, 1] = 1
    rle_str = rle_encode(dummy_mask)
    expected_rle = "2 1 6 1"
    assert (
        rle_str == expected_rle
    ), f"RLE Encoding failed. Got {rle_str}, expected {expected_rle}"
    print("    RLE Encoding: OK")

    # --- Test Utils: Dice Coefficient ---
    t_pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
    t_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Intersection = 1.0
    # Sum = 2 + 2 = 4
    # Dice = (2*1) / 4 = 0.5
    dice = dice_coef(t_pred, t_true)
    assert torch.isclose(
        dice, torch.tensor(0.5), atol=1e-4
    ), f"Dice calculation failed. Got {dice}"
    print("    Dice Coefficient: OK")

    # --- Test Model Architecture ---
    model = MultiTaskResNetFPN().to(Config.DEVICE)
    # Input: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        output = model(dummy_input)

    # Expected Output: (Batch, Classes, Height, Width)
    expected_shape = (2, 2, Config.TILE_SIZE, Config.TILE_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"
    print("    Model Architecture: OK")

    # --- Test Loss Function ---
    criterion = MultiTaskLoss()
    # Create dummy targets (Batch, Classes, Height, Width)
    dummy_targets = torch.randint(0, 2, expected_shape).float().to(Config.DEVICE)
    loss = criterion(output, dummy_targets)

    # Check if loss is scalar and requires grad (if input required grad, but here we used no_grad for model)
    # Let's do a quick backward check with grad enabled
    dummy_input.requires_grad = True
    output_grad = model(dummy_input)
    loss_grad = criterion(output_grad, dummy_targets)
    loss_grad.backward()

    assert not torch.isnan(loss_grad), "Loss returned NaN"
    print("    Loss Function & Backward Pass: OK")

    # ====================================================
    # 3. Training Pipeline
    # ====================================================
    print("\n[3] Executing Training Pipeline (Debug Mode)...")

    trainer = Trainer()

    # Run training
    # This will load metadata, generate tiles (cached in DEMO_WORKDIR), and train for 1 epoch
    trainer.fit(load_cached_data=False)

    # Verify model checkpoint creation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed to save model checkpoint at {Config.MODEL_PATH}"
        )

    print(f"    Training complete. Model saved to {Config.MODEL_PATH}")

    # ====================================================
    # 4. Inference Pipeline
    # ====================================================
    print("\n[4] Executing Inference Pipeline...")

    runner = InferenceRunner()

    # Run inference
    # This will load test metadata, run prediction, and save submission.csv
    runner.run()

    # Verify submission file creation
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to save submission file at {Config.SUBMISSION_PATH}"
        )

    # Validate submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = ["id", "predicted"]
    if not all(col in sub_df.columns for col in required_cols):
        raise ValueError(
            f"Submission file missing required columns. Found: {sub_df.columns}"
        )

    print(f"    Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Submission rows: {len(sub_df)}")
    print(f"    Sample prediction: {sub_df.iloc[0].to_dict()}")

    print("\nAll systems operational. Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
