import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.model_components import ContextGatingBlock, AttentionPooling
from library.model import WhaleConvNeXt
from library.data import get_dataloaders
from library.engine import fit_one_seed, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration of Right Whale Detection Solution ===")

    # 1. Setup & Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast demo
    print("\n[1] Configuring environment for fast demonstration...")

    seed_everything(42)

    # Override Config for Debug/Demo mode
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for quick processing
    Config.EPOCHS = 1  # Single epoch to verify training loop
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.SEEDS = [42]  # Single seed
    Config.PATIENCE = 1  # minimal patience

    # Ensure working directory is clean for this demo run to force computation
    if os.path.exists(Config.WORKING_DIR):
        # We don't delete the whole dir to avoid affecting other runs,
        # but we ensure we are aware of the cache location.
        pass
    else:
        os.makedirs(Config.WORKING_DIR)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Model Components (Unit Tests)
    print("\n[2] Verifying Model Components...")

    # Test ContextGatingBlock
    # Input: (B, 32, 16, 16), Context: (B, 64, 8, 8) -> Output: (B, 32, 16, 16)
    B, C_in, H, W = 2, 32, 16, 16
    C_ctx = 64
    x = torch.randn(B, C_in, H, W)
    ctx = torch.randn(B, C_ctx, H // 2, W // 2)

    gate_block = ContextGatingBlock(in_channels=C_in, context_channels=C_ctx)
    out = gate_block(x, ctx)

    assert out.shape == (
        B,
        C_in,
        H,
        W,
    ), f"ContextGatingBlock output shape mismatch. Expected {(B, C_in, H, W)}, got {out.shape}"
    print("ContextGatingBlock: OK")

    # Test AttentionPooling
    # Input: (B, T, C) -> Output: (B, C)
    T, C = 10, 128
    seq = torch.randn(B, T, C)
    pool_layer = AttentionPooling(input_dim=C)
    out_pool = pool_layer(seq)

    assert out_pool.shape == (
        B,
        C,
    ), f"AttentionPooling output shape mismatch. Expected {(B, C)}, got {out_pool.shape}"
    print("AttentionPooling: OK")

    # Test Full Model Forward Pass
    # Input: (B, 1, F, T) -> Output: (B, 1)
    # Spectrogram dims based on Config: 128 Mels, ~200 Time frames (4000/20)
    F_dim, T_dim = Config.N_MELS, Config.NUM_SAMPLES // Config.HOP_LENGTH
    dummy_spec = torch.randn(B, 1, F_dim, T_dim)

    model = WhaleConvNeXt(
        pretrained=False
    )  # False to avoid downloading weights in demo if not cached
    logits = model(dummy_spec)

    assert logits.shape == (
        B,
        1,
    ), f"WhaleConvNeXt output shape mismatch. Expected {(B, 1)}, got {logits.shape}"
    print("WhaleConvNeXt Forward Pass: OK")

    # 3. Data Pipeline Demonstration
    print("\n[3] Initializing Data Pipeline...")

    # We force load_cached_data=False to demonstrate the processing logic
    # In a real run, this would be True to save time
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Verify Data Loader
    batch_data, batch_targets = next(iter(train_loader))
    print(f"Train Batch Data Shape: {batch_data.shape}")
    print(f"Train Batch Targets Shape: {batch_targets.shape}")

    # Check dimensions: (Batch, 1, Freq, Time)
    expected_shape = (
        Config.BATCH_SIZE,
        1,
        Config.N_MELS,
        Config.NUM_SAMPLES // Config.HOP_LENGTH + 1,
    )
    # Note: +1 or 0 depends on padding/centering in MelSpec.
    # Let's check consistency with the actual output.
    # Config.NUM_SAMPLES = 4000, Hop = 20. 4000/20 = 200.
    # Torchaudio MelSpectrogram with center=True usually gives T = L/Hop + 1.

    assert batch_data.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert (
        batch_data.shape[1] == 1
    ), "Channel dimension mismatch (should be 1 for spectrogram)"
    assert batch_data.shape[2] == Config.N_MELS, "Frequency dimension mismatch"

    print("Data Pipeline: OK")

    # 4. Training Loop Demonstration
    print("\n[4] Running Training Loop (1 Epoch, Seed 42)...")

    # Train for one seed
    model_path, best_auc = fit_one_seed(
        train_loader, val_loader, seed=42, device=Config.DEVICE
    )

    print(f"Training completed. Model saved to: {model_path}")
    print(f"Best Validation AUC: {best_auc:.4f}")

    assert os.path.exists(model_path), "Model file was not saved."
    assert 0 <= best_auc <= 1, "AUC score out of range."

    # 5. Inference & Submission
    print("\n[5] Generating Submission...")

    # Generate submission using the trained model
    # We pass a list of model paths (ensemble support), here just one
    df_submission = generate_submission(test_loader, [model_path], device=Config.DEVICE)

    # Verify Submission
    print("Submission Head:")
    print(df_submission.head())

    expected_rows = len(test_loader.dataset)
    assert (
        len(df_submission) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_submission)}"

    assert list(df_submission.columns) == [
        "clip",
        "probability",
    ], "Submission columns mismatch."

    # Check that probabilities are valid
    probs = df_submission["probability"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range."

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
