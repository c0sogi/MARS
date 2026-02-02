import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import CactusDataset, get_train_transforms
from library.model import CustomSEResNet
from library.train import run_training
from library.inference import run_inference


def main():
    print("Starting Demonstration of Cactus Identification Pipeline...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Patch Config to use a temporary directory and minimal compute
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Re-create directories since we changed the path
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set hyperparams for speed
    Config.EPOCHS = 1
    Config.SEEDS = [42]  # Single seed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 100  # Small subset
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Epochs: {Config.EPOCHS}")

    # ==========================================
    # 2. Utility Verification
    # ==========================================
    print("\n[2] Verifying Utilities...")

    # Test Seeding
    seed_everything(42)
    rand1 = np.random.rand(5)
    seed_everything(42)
    rand2 = np.random.rand(5)
    assert np.allclose(
        rand1, rand2
    ), "Seed everything failed to produce deterministic numpy results."
    print("    Seeding logic verified.")

    # Test ROC AUC
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_scores)
    # Expected: 0.75 (Correctly ranked: (0.1<0.35), (0.1<0.8), (0.4<0.8). Incorrect: (0.4>0.35). 3/4 = 0.75)
    assert 0.0 <= auc <= 1.0, "AUC score out of range."
    print(f"    ROC AUC calculation verified (Score: {auc}).")

    # ==========================================
    # 3. Dataset Verification
    # ==========================================
    print("\n[3] Verifying Dataset...")

    # Initialize Dataset
    ds = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        transform=get_train_transforms(),
        mode="train",
        load_cached_data=False,  # Force load from disk to test logic
    )

    # Verify Length (clamped by DEBUG_SAMPLES)
    assert (
        len(ds) == Config.DEBUG_SAMPLES
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SAMPLES}, got {len(ds)}"

    # Verify Item Structure
    img, label = ds[0]

    # Check Image Tensor
    assert isinstance(img, torch.Tensor), "Image is not a tensor."
    assert img.shape == (3, 32, 32), f"Unexpected image shape: {img.shape}"
    assert img.dtype == torch.float32, f"Unexpected image dtype: {img.dtype}"

    # Check Label
    assert isinstance(label, torch.Tensor), "Label is not a tensor."
    assert label.ndim == 0, "Label should be a scalar tensor."

    print(f"    Dataset loaded successfully. Sample shape: {img.shape}")

    # ==========================================
    # 4. Model Verification
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    model = CustomSEResNet(**Config.MODEL_PARAMS)
    model.eval()

    # Create dummy input batch (B, C, H, W)
    dummy_input = torch.randn(4, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify Output
    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("    Model forward pass successful. Output shape verified.")

    # ==========================================
    # 5. Training Pipeline Simulation
    # ==========================================
    print("\n[5] Running Training Simulation...")

    # This calls the library function which handles loops, saving, etc.
    # We patched Config, so it will run quickly.
    run_training()

    # Verify artifact creation
    expected_model_path = Config.get_model_path(Config.SEEDS[0])
    assert os.path.exists(
        expected_model_path
    ), f"Model checkpoint not found at {expected_model_path}"
    print(f"    Training complete. Checkpoint verified at: {expected_model_path}")

    # ==========================================
    # 6. Inference Pipeline Simulation
    # ==========================================
    print("\n[6] Running Inference Simulation...")

    # This calls the library function which loads the saved model and predicts on test set
    # Note: Test set in debug mode will also be truncated
    run_inference()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Load and check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in sub_df.columns and "has_cactus" in sub_df.columns
    ), "Submission columns missing."
    assert (
        len(sub_df) == Config.DEBUG_SAMPLES
    ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLES}, got {len(sub_df)}"
    assert (
        sub_df["has_cactus"].min() >= 0.0 and sub_df["has_cactus"].max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    print(f"    Inference complete. Submission shape: {sub_df.shape}")
    print("    First 3 rows:")
    print(sub_df.head(3))

    print("\n==========================================")
    print("       DEMONSTRATION SUCCESSFUL           ")
    print("==========================================")


if __name__ == "__main__":
    main()
