import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.dbb_layers import DBBGroupedConv
from library.model import UltraWideDBBResNeXt
from library.engine import run_training_pipeline


def verify_dbb_layer_logic(device):
    """
    Verifies the Diverse Branch Block (DBB) re-parameterization logic.
    Ensures that the fused kernel/bias in deploy mode produces the same output
    as the multi-branch structure in training mode.
    """
    print("Verifying DBB Layer Re-parameterization...")

    # Create a DBB layer with specific config
    # in=16, out=16, k=3, groups=4
    layer = DBBGroupedConv(
        in_channels=16, out_channels=16, kernel_size=3, groups=4, bias=True
    ).to(device)

    layer.eval()

    # Create random input
    input_tensor = torch.randn(2, 16, 32, 32).to(device)

    # 1. Forward pass in 'training' mode (multi-branch)
    with torch.no_grad():
        out_train = layer(input_tensor)

    # 2. Switch to 'deploy' mode (fused single conv)
    layer.switch_to_deploy()

    # 3. Forward pass in 'deploy' mode
    with torch.no_grad():
        out_deploy = layer(input_tensor)

    # 4. Compare outputs
    # Floating point arithmetic may cause slight deviations, so we check against a tolerance
    diff = (out_train - out_deploy).abs().max().item()
    print(f"Max absolute difference between Train and Deploy modes: {diff:.8e}")

    # Assert correctness (tolerance 1e-4 is usually safe for float32)
    assert (
        diff < 1e-4
    ), "DBB Layer re-parameterization failed: Outputs diverge significantly."
    print("DBB Layer logic verified successfully.")


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    # Load with caching disabled to test raw image reading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8, load_cached_data=False, seed=42
    )

    # Fetch a single batch
    images, targets = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Targets Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        8,
        3,
        32,
        32,
    ), "Image batch shape mismatch. Expected (8, 3, 32, 32)."
    assert targets.shape == (8,), "Target batch shape mismatch. Expected (8,)."
    assert images.dtype == torch.float32, "Images should be float32 tensors."
    print("Data Loading verified.")

    # 3. Verify Model Components
    print("\n--- Verifying Model Components ---")

    # Verify DBB Layer logic specifically
    verify_dbb_layer_logic(device)

    # Verify Full Model instantiation and forward pass
    model = UltraWideDBBResNeXt(groups=32).to(device)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), "Model output shape mismatch. Expected (4, 1)."
    print("Model architecture verified.")

    # 4. Execute Training Pipeline
    print("\n--- Executing Training Pipeline ---")
    # We use minimal settings for demonstration speed:
    # 1 Epoch, 1 Seed, small batch size.
    # This utilizes library.engine.run_training_pipeline

    run_training_pipeline(
        epochs=1,
        batch_size=64,
        seeds=[42],  # Single seed for speed
        patience=1,
        load_cached_data=True,  # Use cached data if available (we just cached it above implicitly)
        submission_dir="./submission",
        working_dir="./working/demo_run",
    )

    # 5. Verify Submission Output
    print("\n--- Verifying Submission ---")
    submission_path = "./submission/submission.csv"

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission DataFrame Shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    # Check constraints
    assert (
        df_sub.shape[0] == 3325
    ), f"Submission row count mismatch. Expected 3325, got {df_sub.shape[0]}."
    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns mismatch."
    assert (
        df_sub["has_cactus"].min() >= 0.0 and df_sub["has_cactus"].max() <= 1.0
    ), "Probabilities out of range."

    print("\nAll tasks completed and verified successfully.")


if __name__ == "__main__":
    main()
