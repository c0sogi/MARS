import os
import sys
import numpy as np
import torch
import warnings

# Ensure we can import from the library folder
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import HybridEfficientNet
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Breast Cancer Detection Pipeline Demo ===")

    # 1. Configuration Override for Speed and Demonstration
    print("\n[1] Configuring environment for fast demonstration...")
    seed_everything(42)

    # Modify Config attributes dynamically
    Config.DEBUG = True
    Config.IMG_SIZE = (128, 128)  # Reduce size for faster processing
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.WARMUP_EPOCHS = 0.1  # Ensure pct_start < 1.0 for OneCycleLR
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.BACKBONE = "efficientnet_b0"  # Smaller backbone for speed
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 2. Metric Verification
    print("\n[2] Verifying Metric (Probabilistic F1)...")
    y_true = np.array([1, 0, 1, 0])

    # Case A: Perfect predictions
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = probabilistic_f1(y_true, y_pred_perfect)

    # Case B: Terrible predictions
    y_pred_bad = np.array([0.0, 1.0, 0.0, 1.0])
    score_bad = probabilistic_f1(y_true, y_pred_bad)

    print(f"Score (Perfect): {score_perfect:.4f}")
    print(f"Score (Bad): {score_bad:.4f}")

    assert np.isclose(score_perfect, 1.0), "Metric failed: Perfect score should be 1.0"
    assert score_bad < 0.1, "Metric failed: Bad score should be near 0"
    print("Metric verification passed.")

    # 3. Data Pipeline Verification
    print("\n[3] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch a single batch from training loader
    images, tab_features, targets = next(iter(train_loader))

    print(f"Image Batch Shape: {images.shape}")
    print(f"Tabular Batch Shape: {tab_features.shape}")
    print(f"Targets Batch Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Image tensor shape mismatch"
    assert tab_features.ndim == 2, "Tabular features should be 2D (Batch, Features)"
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"
    print("Data loading verification passed.")

    # 4. Model Verification
    print("\n[4] Verifying Model Initialization and Inference...")
    tabular_input_dim = tab_features.shape[1]

    model = HybridEfficientNet(tabular_input_dim=tabular_input_dim)
    model = model.to(Config.DEVICE)

    # Run forward pass
    with torch.no_grad():
        images = images.to(Config.DEVICE)
        tab_features = tab_features.to(Config.DEVICE)
        logits = model(images, tab_features)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.BATCH_SIZE, 1), "Output logits shape mismatch"
    print("Model inference verification passed.")

    # 5. Training Loop Simulation
    print("\n[5] Running Training Loop (1 Epoch)...")

    # run_training uses the Config settings we modified earlier
    run_training(debug=True)

    # Verify artifact creation
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model successfully saved to {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Training completed but model file was not found.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
