import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Add the current directory to path to ensure library imports work if not already set
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.layers import SwiGLU, PreNormResBlock, TransitionLayer
from library.model import ContextAwareSwishGatedResFunnel
from library.train import run_training

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print(
        "=== Starting Demonstration of Manufacturing Control Prediction Pipeline ===\n"
    )

    # 1. Setup
    print("[Step 1] Setting up environment and seeds...")
    set_seed(42)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n[Step 2] Verifying Data Loading (using 5% data fraction)...")
    # We use a small fraction to speed up the demo
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=128, data_fraction=0.05, load_cached_data=True
    )

    # Fetch one batch to verify structure
    x_cont_batch, x_cat_batch, y_batch = next(iter(train_loader))

    print(
        f"Batch shapes -> Continuous: {x_cont_batch.shape}, Categorical: {x_cat_batch.shape}, Target: {y_batch.shape}"
    )

    # Assertions
    assert (
        x_cont_batch.shape[1] == 30
    ), f"Expected 30 continuous features, got {x_cont_batch.shape[1]}"
    assert (
        x_cat_batch.shape[1] == 10
    ), f"Expected 10 categorical features (chars), got {x_cat_batch.shape[1]}"
    assert y_batch.ndim == 1, "Target should be a 1D tensor"
    assert x_cont_batch.dtype == torch.float32, "Continuous features should be float32"
    assert x_cat_batch.dtype == torch.int64, "Categorical features should be int64"

    print("Data Loader verification passed.")

    # 3. Layer Verification
    print("\n[Step 3] Verifying Custom Layers...")

    # Test SwiGLU
    # SwiGLU expects input dim -> splits to dim/2.
    # The provided implementation: x.chunk(2, dim=-1). So input size must be even.
    dim = 32
    swiglu = SwiGLU()
    dummy_input = torch.randn(10, dim * 2)  # Input is 2*dim
    output = swiglu(dummy_input)
    assert output.shape == (
        10,
        dim,
    ), f"SwiGLU output shape mismatch. Expected (10, {dim}), got {output.shape}"
    print("SwiGLU layer verified.")

    # Test PreNormResBlock
    # This block projects dim -> 2*dim internally before SwiGLU
    block = PreNormResBlock(dim=dim, dropout=0.1, drop_path=0.0)
    dummy_input = torch.randn(10, dim)
    output = block(dummy_input)
    assert output.shape == (
        10,
        dim,
    ), f"PreNormResBlock output shape mismatch. Expected (10, {dim}), got {output.shape}"
    print("PreNormResBlock verified.")

    # 4. Model Architecture Verification
    print("\n[Step 4] Verifying Full Model Architecture...")
    model = ContextAwareSwishGatedResFunnel().to(device)

    # Move batch to device
    x_cont_dev = x_cont_batch.to(device)
    x_cat_dev = x_cat_batch.to(device)

    # Forward pass
    logits = model(x_cont_dev, x_cat_dev)

    # Check output
    assert logits.shape == (
        128,
        1,
    ), f"Model output shape mismatch. Expected (128, 1), got {logits.shape}"
    print("Model forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n[Step 5] Executing Training Loop (1 Epoch, 5% Data)...")

    # We use the run_training function from library.train
    # It handles optimizer creation, training loop, validation, and saving the best model.
    best_auc = run_training(data_fraction=0.05, epochs=1)

    print(f"Training finished. Best Validation AUC: {best_auc:.4f}")

    # Validate that the model file was created
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert 0.0 <= best_auc <= 1.0, "AUC score out of valid range"
    print("Training loop verified and model saved.")

    # 6. Inference Demonstration
    print("\n[Step 6] Demonstrating Inference with Saved Model...")

    # Load the model state
    loaded_model = ContextAwareSwishGatedResFunnel().to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    # Get a test batch
    x_test_cont, x_test_cat, _ = next(iter(test_loader))
    x_test_cont = x_test_cont.to(device)
    x_test_cat = x_test_cat.to(device)

    with torch.no_grad():
        test_logits = loaded_model(x_test_cont, x_test_cat).squeeze()
        test_probs = torch.sigmoid(test_logits)

    print(f"Generated predictions for batch of size {len(test_probs)}")
    print(f"Sample predictions: {test_probs[:5].cpu().numpy()}")

    assert (
        test_probs.min() >= 0.0 and test_probs.max() <= 1.0
    ), "Probabilities must be between 0 and 1"
    print("Inference verification passed.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
