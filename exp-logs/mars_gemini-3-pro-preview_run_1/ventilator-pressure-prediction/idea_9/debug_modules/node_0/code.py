import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import provided library components
from library.config import Config
from library.utils import set_seed
from library.features import FeatureEngineer
from library.data_loader import get_data_loaders
from library.model import PhysicsInjectedNet
from library.train import Trainer


def run_demo():
    print("=== Ventilator Pressure Prediction: Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1/6] Configuring environment for rapid demonstration...")

    # Define a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch the Config class to optimize for speed and use demo paths
    # Note: We must update CACHE_DIR explicitly as it was assigned at import time
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Enable Debug mode to sample a small subset of breaths (e.g., 200)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200

    # Reduce training parameters for the demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small demo payload

    # Set reproducibility
    set_seed(42)
    print(f"      Working Directory: {Config.WORKING_DIR}")
    print(f"      Debug Mode: {Config.DEBUG} (Sample Size: {Config.DEBUG_SAMPLE_SIZE})")
    print(f"      Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    print("\n[2/6] Executing Feature Engineering...")

    # Instantiate engine and force processing (load_cached=False ensures we generate the debug subset)
    engineer = FeatureEngineer()
    data = engineer.load_data(load_cached=False)

    # Extract artifacts for validation
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]

    print(f"      Generated Train X shape: {train_x.shape}")
    print(f"      Generated Train y shape: {train_y.shape}")

    # Logic Verification
    # Shape should be (N_breaths, 80, N_features)
    assert train_x.ndim == 3, "Train X must be 3D numpy array"
    assert (
        train_x.shape[1] == Config.N_STEPS
    ), f"Sequence length must be {Config.N_STEPS}"
    assert (
        train_x.shape[0] == train_y.shape[0]
    ), "Mismatch between X and y breath counts"
    assert train_y.shape[1] == Config.N_STEPS, "Target sequence length mismatch"

    # Check if physics features (e.g., volume, R_u_in) were actually computed
    # Config.FEATURE_COLS defines the order. Let's check the last dimension size matches config
    assert train_x.shape[2] == len(
        Config.FEATURE_COLS
    ), "Feature dimension mismatch with Config"

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("\n[3/6] Initializing Data Loaders...")

    # Retrieve loaders using the cached data we just generated
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached=True
    )

    # Fetch one batch to verify tensor conversion
    batch_x, batch_y = next(iter(train_loader))

    print(f"      Batch X Tensor shape: {batch_x.shape}")
    print(f"      Batch y Tensor shape: {batch_y.shape}")

    assert isinstance(batch_x, torch.Tensor), "DataLoader should yield Tensors"
    assert batch_x.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_x.dtype == torch.float32, "Input tensor should be float32"

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4/6] Initializing Physics-Injected Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInjectedNet().to(device)

    print(f"      Device: {device}")
    print(f"      Model Architecture: {type(model).__name__}")

    # Forward Pass Verification
    batch_x = batch_x.to(device)
    with torch.no_grad():
        preds = model(batch_x)

    print(f"      Forward pass output shape: {preds.shape}")

    # Output should be (Batch, Seq_Len)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.N_STEPS,
    ), "Model output shape mismatch"

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5/6] Running Training Loop...")

    trainer = Trainer(train_loader, val_loader)

    # Run fit (patience=1 for demo purposes)
    trainer.fit(patience=1)

    # Verify model checkpoint creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"
    print("      Training completed and model saved successfully.")

    # ---------------------------------------------------------
    # 6. Inference Simulation
    # ---------------------------------------------------------
    print("\n[6/6] Simulating Inference on Test Set...")

    # Load the best saved model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Run inference on a test batch
    test_batch_x = next(iter(test_loader))
    test_batch_x = test_batch_x.to(device)

    with torch.no_grad():
        test_preds = model(test_batch_x)

    print(f"      Test predictions shape: {test_preds.shape}")
    print(f"      Sample prediction (first 5 steps): {test_preds[0, :5].cpu().numpy()}")

    assert test_preds.shape[1] == Config.N_STEPS, "Inference sequence length mismatch"

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
