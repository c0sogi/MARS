import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Set random seeds for reproducibility
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Import provided library modules
from library.config import Config
from library.custom_layers import DualPooling, CBAM, TriPathReadout
from library.data_processing import (
    load_data,
    IcebergDataset,
    get_fold_loaders,
    get_test_loader,
)
from library.model_architecture import TriPathWideBodyNet
from library.training_engine import train_fold


def run_demo():
    print("============================================================")
    print("   Iceberg Classification Solution - Demonstration Script   ")
    print("============================================================")

    # 1. Override Config for Speed
    # We modify the Config class attributes directly to run a minimal version of the pipeline.
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.NUM_FOLDS = 2  # Minimum folds for split
    Config.BATCH_SIZE = 4  # Small batch size
    Config.PATIENCE = 1  # Aggressive early stopping
    Config.WORK_DIR = "./working/demo_run"  # Separate working dir for demo
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")

    # Ensure clean slate for demo
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    Config.setup()

    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Working Directory: {Config.WORK_DIR}")

    # 2. Data Processing Verification
    print("\n[2] Verifying Data Processing...")

    # Load data (this will trigger processing from scratch or load cache if available)
    data_dict = load_data(load_cached_data=False)  # Force process to demo logic

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    inc_train = data_dict["inc_train"]

    # Assertions for data shapes
    print("    Verifying data shapes...")
    assert (
        len(X_train) == len(y_train) == len(inc_train)
    ), "Mismatch in training data lengths"
    assert X_train.shape[1:] == (
        75,
        75,
        3,
    ), f"Incorrect image shape: {X_train.shape[1:]}"
    assert not np.isnan(
        inc_train
    ).any(), "Incidence angles contain NaNs after processing"
    print(f"    Data loaded successfully. Train samples: {len(X_train)}")

    # Test Dataset Class
    print("    Verifying IcebergDataset...")
    ds = IcebergDataset(
        X_train[:10],
        inc_train[:10],
        y_train[:10],
        data_dict["min_vals"],
        data_dict["max_vals"],
        transform=True,
    )
    img, inc, lbl = ds[0]

    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    assert img.shape == (3, 75, 75), f"Incorrect tensor shape: {img.shape}"
    assert isinstance(lbl, torch.Tensor), "Label is not a Tensor"
    print("    IcebergDataset returns correct types and shapes.")

    # Test Data Loaders
    print("    Verifying DataLoader generation...")
    train_loader, val_loader = get_fold_loaders(0, load_cached_data=True)
    batch_img, batch_inc, batch_lbl = next(iter(train_loader))
    assert batch_img.shape[0] == Config.BATCH_SIZE, "DataLoader batch size mismatch"
    print("    DataLoaders created successfully.")

    # 3. Component Verification
    print("\n[3] Verifying Custom Layers...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(2, 64, 75, 75).to(device)  # Batch=2, Channels=64, 75x75

    # Test DualPooling
    # Input (B, C, H, W) -> Output (B, 2*C, H/2, W/2)
    dp = DualPooling(kernel_size=2, stride=2).to(device)
    out_dp = dp(dummy_input)
    expected_shape = (2, 128, 37, 37)  # 75//2 = 37
    assert (
        out_dp.shape == expected_shape
    ), f"DualPooling output shape mismatch. Got {out_dp.shape}, expected {expected_shape}"
    print("    DualPooling: OK")

    # Test CBAM
    # Input (B, C, H, W) -> Output (B, C, H, W)
    cbam = CBAM(in_channels=64).to(device)
    out_cbam = cbam(dummy_input)
    assert out_cbam.shape == dummy_input.shape, "CBAM altered output shape unexpectedly"
    print("    CBAM: OK")

    # Test TriPathReadout
    # Input needs to be what the backbone outputs before readout.
    # Let's assume backbone output is (B, 256, 4, 4) based on model architecture comments
    dummy_feat = torch.randn(2, 256, 4, 4).to(device)
    readout = TriPathReadout(in_channels=256, path_a_out_channels=48).to(device)
    out_readout = readout(dummy_feat)
    # Path A: 48 * 4 * 4 = 768
    # Path B: 256
    # Path C: 256
    # Total: 1280
    assert out_readout.shape == (
        2,
        1280,
    ), f"Readout output shape mismatch. Got {out_readout.shape}"
    print("    TriPathReadout: OK")

    # 4. Model Architecture Verification
    print("\n[4] Verifying Full Model Architecture...")
    model = TriPathWideBodyNet().to(device)

    # Input: (B, 3, 75, 75) and (B,)
    dummy_img = torch.randn(Config.BATCH_SIZE, 3, 75, 75).to(device)
    dummy_inc = torch.randn(Config.BATCH_SIZE).to(device)

    output = model(dummy_img, dummy_inc)

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Got {output.shape}"
    print("    TriPathWideBodyNet forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n[5] Demonstrating Training Loop (Fold 0)...")
    val_loss = train_fold(0)

    print(f"    Training complete. Validation Loss: {val_loss:.4f}")

    # Verify model artifact
    model_path = Config.get_model_path(0)
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
    print(f"    Model checkpoint verified at: {model_path}")

    # 6. Inference Demonstration
    print("\n[6] Demonstrating Inference...")

    # Load the model state
    model = TriPathWideBodyNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get test data
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # Run inference on one batch
    with torch.no_grad():
        imgs, incs = next(iter(test_loader))
        imgs = imgs.to(device)
        incs = incs.to(device)

        logits = model(imgs, incs)
        probs = torch.sigmoid(logits)

    print(f"    Inference successful on batch of size {len(probs)}")
    print(f"    Sample Predictions (ID: Probability):")
    for i in range(min(3, len(probs))):
        print(f"      {test_ids[i]}: {probs[i].item():.4f}")

    # Verify probabilities range
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    print("\n============================================================")
    print("   Demonstration Completed Successfully                     ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
