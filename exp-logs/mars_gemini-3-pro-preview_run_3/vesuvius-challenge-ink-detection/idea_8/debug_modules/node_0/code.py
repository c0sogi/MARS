import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Import library modules
from library.config import Config
from library.utils import seed_everything, rle_encode, fbeta_score
from library.model import FRDUNet
from library.losses import BCEDiceLoss
from library.train import train_model
from library.inference import generate_submission

if __name__ == "__main__":
    # 1. Setup and Configuration
    # --------------------------
    warnings.filterwarnings("ignore")

    # Define directories
    DEMO_DIR = Path("./working/demo_execution")
    CACHE_SOURCE_DIR = Path("./working/demo_cache")
    STATS_SOURCE_DIR = Path("./working/idea_1")  # Known location from file listing

    # Clean and create demo directory
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)

    print("--- Setting up Demo Environment ---")

    # Symlink cached data to avoid expensive loading of 3D volumes
    # We look for .npy files in the cache source
    if CACHE_SOURCE_DIR.exists():
        print(f"Linking cached data from {CACHE_SOURCE_DIR}...")
        for file_path in CACHE_SOURCE_DIR.glob("*.npy"):
            dest_path = DEMO_DIR / file_path.name
            if not dest_path.exists():
                os.symlink(file_path.resolve(), dest_path)

    # Copy normalization stats if available to skip computation
    stats_file = STATS_SOURCE_DIR / "normalization_stats.npy"
    if stats_file.exists():
        shutil.copy(stats_file, DEMO_DIR / "normalization_stats.npy")
        print("Copied normalization stats.")

    # Override Config parameters for speed
    print("Overriding configuration for fast execution...")
    Config.WORKING_DIR = DEMO_DIR
    Config.BEST_MODEL_PATH = DEMO_DIR / "best_model.pth"
    Config.SUBMISSION_PATH = Path("./submission.csv")

    # Hyperparameters for demo
    Config.NUM_EPOCHS = 1
    Config.TRAIN_SAMPLES_PER_EPOCH = 16  # Process only 16 patches per epoch
    Config.BATCH_SIZE = 4
    Config.VAL_STRIDE = 512  # Large stride to speed up validation inference
    Config.TTA_ENABLED = False  # Disable TTA for speed

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # 2. Verify Utility Functions
    # ---------------------------
    print("\n--- Verifying Utilities ---")

    # Test RLE Encoding
    # Mask: [[0, 1, 1, 0]] -> Flattened: 0, 1, 1, 0
    # 1-based indexing: Pixel 2 is 1, Pixel 3 is 1. Run starts at 2, length 2.
    dummy_mask = np.array([[0, 1, 1, 0]], dtype=np.uint8)
    encoded = rle_encode(dummy_mask)
    assert encoded == "2 2", f"RLE verification failed. Expected '2 2', got '{encoded}'"
    print("RLE encoding verified.")

    # Test F-beta Score
    # Pred: [1, 0], Target: [1, 1]. TP=1, FN=1, FP=0.
    # Precision=1.0, Recall=0.5. F0.5 should be < 1.0 but > 0.5.
    # Formula: (1.25 * 1 * 0.5) / (0.25 * 1 + 0.5) = 0.625 / 0.75 = 0.8333
    y_p = np.array([1, 0])
    y_t = np.array([1, 1])
    score = fbeta_score(y_p, y_t, beta=0.5)
    assert 0.8 < score < 0.9, f"F-beta verification failed. Got {score}"
    print("F-beta score verified.")

    # 3. Verify Model Logic
    # ---------------------
    print("\n--- Verifying Model Architecture ---")
    device = torch.device(Config.DEVICE)
    model = FRDUNet().to(device)

    # Create dummy input matching Config.Z_DEPTH (65)
    # Shape: (Batch, Z, H, W)
    dummy_input = torch.randn(2, 65, 128, 128).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch, 1, H, W)
    assert output.shape == (
        2,
        1,
        128,
        128,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass verified.")

    # 4. Verify Loss Function
    # -----------------------
    print("\n--- Verifying Loss Function ---")
    criterion = BCEDiceLoss()
    dummy_target = torch.randint(0, 2, (2, 128, 128)).float().to(device)

    # Loss expects logits (output) and targets
    # Output from model is (B, 1, H, W), target is (B, H, W)
    loss = criterion(output.squeeze(1), dummy_target)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"Loss calculation verified. Value: {loss.item():.4f}")

    # 5. Run Training
    # ---------------
    print("\n--- Starting Training (Demo) ---")
    # This runs the training loop using the overridden Config
    try:
        train_model(load_cached_data=True)
    except Exception as e:
        print(f"Training failed: {e}")
        raise e

    if not Config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Training completed but best_model.pth was not created."
        )
    print("Training completed successfully.")

    # 6. Run Inference
    # ----------------
    print("\n--- Starting Inference (Demo) ---")
    # This generates submission.csv
    try:
        generate_submission(load_cached_data=True)
    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

    if not Config.SUBMISSION_PATH.exists():
        raise FileNotFoundError(
            "Inference completed but submission.csv was not created."
        )

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    if list(df_sub.columns) != ["Id", "Predicted"]:
        raise ValueError(f"Invalid submission columns: {df_sub.columns}")
    if len(df_sub) == 0:
        raise ValueError("Submission file is empty.")

    print(f"Inference completed. Submission generated at {Config.SUBMISSION_PATH}")
    print("\nAll tasks executed successfully.")
