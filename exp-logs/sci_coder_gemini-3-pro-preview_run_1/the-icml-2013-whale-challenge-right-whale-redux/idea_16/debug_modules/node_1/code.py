import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, calculate_auc, mixup_data, mixup_criterion
from library.dataset import get_dataloaders, WhaleDataset
from library.layers import (
    CoordinateAttention,
    ContextGatedSpectralPooling,
    ContextGatedResNet18,
)
from library.trainer import run_training_and_submission


def demo_utils():
    """
    Demonstrates and validates utility functions: AUC calculation and Mixup augmentation.
    """
    print("\n=== Demonstrating Utils ===")

    # 1. Test AUC Calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_auc(y_true, y_pred)
    print(f"Calculated AUC: {auc}")
    assert 0 <= auc <= 1, "AUC should be between 0 and 1"

    # 2. Test Mixup Augmentation
    print("Testing Mixup...")
    batch_size = 4
    channels = 1
    freq = 128
    time_steps = 125

    # Create dummy batch
    x = torch.randn(batch_size, channels, freq, time_steps)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    # Apply Mixup
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)

    # Verify shapes and logic
    assert mixed_x.shape == x.shape, "Mixed input shape mismatch"
    assert y_a.shape == y.shape, "Target A shape mismatch"
    assert y_b.shape == y.shape, "Target B shape mismatch"
    assert 0 <= lam <= 1, "Lambda should be between 0 and 1"
    print("Mixup utils verified.")


def demo_dataset_and_loader():
    """
    Demonstrates loading the dataset and creating dataloaders.
    Uses debug=True to load a tiny subset and avoid large-scale processing.
    """
    print("\n=== Demonstrating Dataset & DataLoader (Debug Mode) ===")

    # Load dataloaders in debug mode (no caching to ensure fresh processing)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch from training loader
    inputs, labels, clips = next(iter(train_loader))

    print(f"Input batch shape: {inputs.shape}")
    print(f"Labels batch shape: {labels.shape}")

    # Assertions to verify data pipeline
    # Shape: (Batch, Channel, Freq, Time)
    assert inputs.dim() == 4, "Input must be 4D (B, C, F, T)"
    assert inputs.shape[1] == 1, "Channel dim should be 1 (Mono)"
    assert inputs.shape[2] == Config.N_MELS, f"Freq dim should be {Config.N_MELS}"
    assert inputs.shape[3] > 0, "Time dim should be positive"

    assert labels.dim() == 1, "Labels must be 1D"
    assert len(clips) == inputs.shape[0], "Clips count must match batch size"

    return inputs  # Return batch for model testing


def demo_layers_and_model(dummy_input):
    """
    Demonstrates and validates the custom layers and the full model.
    """
    print("\n=== Demonstrating Layers & Model ===")
    device = torch.device("cpu")

    # 1. Test Coordinate Attention Block
    print("Testing CoordinateAttention...")
    # Dummy input: (B, C, H, W)
    ca_input = torch.randn(2, 16, 32, 32)
    ca_block = CoordinateAttention(inp=16, reduction=4)
    ca_out = ca_block(ca_input)
    assert ca_out.shape == ca_input.shape, "CoordinateAttention output shape mismatch"
    print("CoordinateAttention verified.")

    # 2. Test Context-Gated Spectral Pooling
    print("Testing ContextGatedSpectralPooling...")
    # Dimensions simulate outputs from ResNet layers (L2, L3, L4)
    B = 2
    T = 10
    x2 = torch.randn(B, 128, 32, T)  # Layer 2
    x3 = torch.randn(B, 256, 16, T)  # Layer 3
    x4 = torch.randn(B, 512, 8, T)  # Layer 4

    cg_pool = ContextGatedSpectralPooling(c2=128, f2=32, c3=256, f3=16, c4=512)
    pool_out = cg_pool(x2, x3, x4)

    # Expected Output: (B, 256, T) after fusion and SE
    assert pool_out.shape == (
        B,
        256,
        T,
    ), f"CG Pooling output shape mismatch: {pool_out.shape}"
    print("ContextGatedSpectralPooling verified.")

    # 3. Test Full ContextGatedResNet18 Model
    print("Testing ContextGatedResNet18...")
    model = ContextGatedResNet18(config=Config)
    model.eval()

    # Use the real spectrogram batch from the dataset demo
    model_input = dummy_input.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(model_input)

    print(f"Model output shape: {logits.shape}")
    # Output should be (Batch_Size, 1) logits
    assert logits.shape == (model_input.shape[0], 1), "Model output should be (B, 1)"
    print("Full model forward pass verified.")


def demo_training_execution():
    """
    Demonstrates the full training and submission pipeline using the trainer module.
    Runs in debug mode for speed (fewer epochs, small dataset).
    """
    print("\n=== Demonstrating Full Training Pipeline (Debug) ===")

    # Ensure working directory is clean for this run logic if needed,
    # but the trainer handles overwrites.

    # Execute training and submission generation
    # debug=True forces:
    # 1. Small dataset subset
    # 2. Fewer epochs (2 epochs)
    # 3. No caching of intermediate data
    try:
        run_training_and_submission(load_cached_data=False, debug=True)
    except Exception as e:
        print(f"Training pipeline failed with error: {e}")
        raise e

    # Verify the output submission file
    if os.path.exists(Config.SUBMISSION_FILE):
        df = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"Submission file created at {Config.SUBMISSION_FILE}")
        print(f"Submission rows: {len(df)}")

        assert len(df) > 0, "Submission file is empty"
        assert (
            "clip" in df.columns and "probability" in df.columns
        ), "Submission columns mismatch"

        # In debug mode, we only process a small subset of test data (50 samples)
        # So we expect 50 rows.
        assert len(df) == 50, f"Expected 50 rows in debug mode, found {len(df)}"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # Configuration
    warnings.filterwarnings("ignore")
    set_seed(42)

    # Step 1: Verify Utilities
    demo_utils()

    # Step 2: Verify Dataset Loading and Processing
    sample_batch = demo_dataset_and_loader()

    # Step 3: Verify Model Architecture
    demo_layers_and_model(sample_batch)

    # Step 4: Verify Full Training Loop and Inference
    demo_training_execution()

    print("\nAll demonstrations completed successfully.")
