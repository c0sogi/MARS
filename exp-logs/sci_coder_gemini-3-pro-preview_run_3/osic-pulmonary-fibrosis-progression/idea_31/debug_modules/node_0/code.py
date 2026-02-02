import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, InverseScaler, LaplaceMetric
from library.data import get_dataloaders
from library.model import ZIMARNet
from library.train import Trainer
from library.inference import generate_submission


def run_demo():
    print("Starting ZIMAR-Net Library Demonstration...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True  # Uses a small subset of data
    Config.DEBUG_SAMPLE_SIZE = 10  # Very small subset for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Redirect working directories to a demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Set seed
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # =========================================================================
    # 2. Verify Utilities
    # =========================================================================
    print("\n[2] Verifying Utility Classes...")

    # --- Test InverseScaler ---
    # Mock training metadata stats: Mean=2000, Std=500
    # We can't easily mock the file read inside __init__, so we rely on the actual metadata file
    # existing at Config.TRAIN_CSV (which is provided in the environment).
    scaler = InverseScaler()
    print(
        f"    InverseScaler loaded stats: Mean={scaler.mean:.2f}, Std={scaler.std:.2f}"
    )

    # Test calculation: z=1.0 should be mean + 1*std
    z_val = torch.tensor([1.0])
    z_sigma = torch.tensor([0.5])
    fvc_raw, sigma_raw = scaler(z_val, z_sigma)

    expected_fvc = scaler.mean + scaler.std
    expected_sigma = 0.5 * scaler.std

    assert np.isclose(
        fvc_raw.item(), expected_fvc, rtol=1e-5
    ), "InverseScaler FVC calculation failed"
    assert np.isclose(
        sigma_raw.item(), expected_sigma, rtol=1e-5
    ), "InverseScaler Sigma calculation failed"
    print("    InverseScaler logic verified.")

    # --- Test LaplaceMetric ---
    metric = LaplaceMetric()

    # Case: Perfect prediction (Delta=0), Sigma=100 (>70 clipped)
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100) = -ln(141.42) approx -4.95
    pred_fvc = np.array([2500.0])
    pred_sigma = np.array([100.0])
    true_fvc = np.array([2500.0])

    metric.update(pred_fvc, pred_sigma, true_fvc)
    score = metric.compute()

    expected_score = -np.log(np.sqrt(2) * 100)
    assert np.isclose(
        score, expected_score, rtol=1e-4
    ), f"LaplaceMetric failed. Got {score}, expected {expected_score}"
    print("    LaplaceMetric logic verified.")

    # =========================================================================
    # 3. Verify Data Pipeline
    # =========================================================================
    print("\n[3] Verifying Data Pipeline (DataLoaders & Preprocessing)...")

    # This triggers image caching and dataset creation
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Check keys
    required_keys = ["image", "clinical", "target", "fvc_raw", "patient_week"]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Check shapes
    # Image: (B, 3, 260, 260)
    imgs = batch["image"]
    assert imgs.dim() == 4, f"Image tensor dimension error. Got {imgs.dim()}"
    assert imgs.shape[1] == 3, f"Image channel error. Got {imgs.shape[1]}"
    assert (
        imgs.shape[2] == Config.IMAGE_SIZE
    ), f"Image height error. Got {imgs.shape[2]}"

    # Clinical: (B, 5) -> [Baseline_FVC, Time, Age, Sex, Smoking]
    clinical = batch["clinical"]
    assert clinical.shape[1] == 5, f"Clinical vector shape error. Got {clinical.shape}"

    print(
        f"    Batch shapes verified: Image {tuple(imgs.shape)}, Clinical {tuple(clinical.shape)}"
    )

    # =========================================================================
    # 4. Verify Model Architecture
    # =========================================================================
    print("\n[4] Verifying ZIMARNet Architecture...")

    model = ZIMARNet()
    model.eval()

    # Forward pass with batch from data loader
    with torch.no_grad():
        mu, sigma = model(imgs, clinical)

    # Check output shapes (B,)
    assert mu.shape[0] == imgs.shape[0], "Output batch size mismatch"
    assert sigma.shape[0] == imgs.shape[0], "Output batch size mismatch"

    # Check Sigma positivity (Softplus + epsilon)
    assert (sigma > 0).all(), "Sigma must be positive"

    # Check Zero-Initialization of Stream B
    # The last linear layer of visual_mlp should have 0 weights and bias
    last_visual_layer = model.visual_mlp[2]
    w_sum = last_visual_layer.weight.abs().sum().item()
    b_sum = last_visual_layer.bias.abs().sum().item()

    assert w_sum == 0.0 and b_sum == 0.0, "Visual Stream Zero-Initialization failed"
    print("    Model Forward Pass & Zero-Init verified.")

    # =========================================================================
    # 5. Demonstrate Training Loop
    # =========================================================================
    print("\n[5] Demonstrating Training Loop (1 Epoch)...")

    trainer = Trainer(debug=True)

    # Run fit (this will run train_epoch and validate)
    trainer.fit()

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print(f"    Training complete. Checkpoint saved at {Config.BEST_MODEL_PATH}")

    # =========================================================================
    # 6. Demonstrate Inference
    # =========================================================================
    print("\n[6] Demonstrating Inference & Submission Generation...")

    # Generate submission using the trained model
    generate_submission(debug=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(sub_df)} rows.")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check values
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"
    assert (
        sub_df["Confidence"] >= 70
    ).all(), "Confidence values below 70ml threshold found (clipping failed)"

    print("    Submission format verified.")

    print("\n=========================================================================")
    print("ZIMAR-Net Library Demonstration Completed Successfully.")
    print("=========================================================================")


if __name__ == "__main__":
    run_demo()
