import os
import sys
import numpy as np
import torch
import pandas as pd
import warnings

# ------------------------------------------------------------------------------
# 1. Setup & Patching
# ------------------------------------------------------------------------------
# Suppress warnings
warnings.filterwarnings("ignore")


# Patch tqdm to prevent progress bars from printing, as per requirements
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


# We need to patch tqdm in the modules that use it.
# Importing them first to ensure they are loaded, then patching.
import library.inference

library.inference.tqdm = silent_tqdm

# Import library components after patching
from library.config import Config
from library.utils import seed_everything
from library.layers import RMSNorm, SwiGLU, RoPEAttention, RoPETransformerEncoderLayer
from library.model import RoPESwiGLURMSNet
from library.data import Tokenizer, get_dataloaders
from library.train import run_training
from library.inference import predict


def main():
    print("=== Starting Demonstration Script ===")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Ensure working directory exists (Config.setup() does this, but good to be sure)
    Config.setup()

    # --------------------------------------------------------------------------
    # 2. Verify Layers (Unit Testing)
    # --------------------------------------------------------------------------
    print("\n[1/5] Verifying Layers...")

    batch_size = 4
    seq_len = 10
    embed_dim = 32

    # Test RMSNorm
    rms = RMSNorm(dim=embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    y = rms(x)
    assert y.shape == x.shape, f"RMSNorm output shape mismatch: {y.shape}"
    # RMSNorm should result in roughly unit variance (ignoring epsilon/affine)
    # We just check it runs and returns valid values (no NaNs)
    assert not torch.isnan(y).any(), "RMSNorm produced NaNs"
    print("  - RMSNorm: OK")

    # Test SwiGLU
    # SwiGLU expects input dim to be 2 * output dim because it chunks input
    swiglu = SwiGLU()
    x_swiglu = torch.randn(batch_size, seq_len, embed_dim * 2)
    y_swiglu = swiglu(x_swiglu)
    assert y_swiglu.shape == (
        batch_size,
        seq_len,
        embed_dim,
    ), f"SwiGLU output shape mismatch: {y_swiglu.shape}"
    print("  - SwiGLU: OK")

    # Test RoPEAttention
    rope_attn = RoPEAttention(embed_dim=embed_dim, num_heads=4)
    x_attn = torch.randn(batch_size, seq_len, embed_dim)
    y_attn = rope_attn(x_attn)
    assert (
        y_attn.shape == x_attn.shape
    ), f"RoPEAttention output shape mismatch: {y_attn.shape}"
    print("  - RoPEAttention: OK")

    # --------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[2/5] Verifying Model Architecture...")

    model = RoPESwiGLURMSNet()
    model.eval()

    # Create dummy inputs
    # Categorical: (Batch, Seq_Len) with indices 0-25
    dummy_cat = torch.randint(0, 26, (batch_size, Config.SEQUENCE_LENGTH))
    # Continuous: (Batch, Num_Cont_Features)
    dummy_cont = torch.randn(batch_size, Config.NUM_CONT_FEATURES)

    with torch.no_grad():
        output = model(dummy_cat, dummy_cont)

    # Expected output: (Batch, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch: {output.shape}"
    print(f"  - Forward pass successful. Output shape: {output.shape}")

    # --------------------------------------------------------------------------
    # 4. Verify Data Processing
    # --------------------------------------------------------------------------
    print("\n[3/5] Verifying Data Processing...")

    # Test Tokenizer
    tokenizer = Tokenizer()
    sample_series = pd.Series(["ABCDEFGHIJ", "ZZZZZZZZZZ"])
    tokens = tokenizer.transform(sample_series)

    expected_0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # A-J
    expected_1 = np.array([25] * 10)  # Z

    assert np.array_equal(tokens[0], expected_0), "Tokenizer failed on A-J"
    assert np.array_equal(tokens[1], expected_1), "Tokenizer failed on Z"
    print("  - Tokenizer: OK")

    # Test DataLoaders (using debug mode for speed)
    print("  - Loading DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    x_cat_batch, x_cont_batch, y_batch = next(iter(train_loader))

    assert x_cat_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQUENCE_LENGTH,
    ), "Train batch x_cat shape mismatch"
    assert x_cont_batch.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CONT_FEATURES,
    ), "Train batch x_cont shape mismatch"
    assert y_batch.shape == (Config.BATCH_SIZE, 1), "Train batch y shape mismatch"
    print("  - DataLoaders: OK")

    # --------------------------------------------------------------------------
    # 5. Verify Training Pipeline
    # --------------------------------------------------------------------------
    print("\n[4/5] Running Training (Debug Mode)...")

    # Run training with debug=True (runs for 2 epochs on 10k samples)
    best_auc = run_training(debug=True, load_cached_data=True)

    # Check if model file was created
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH} after training."
        )

    print(f"  - Training complete. Best AUC: {best_auc:.4f}")
    print(f"  - Model saved to: {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 6. Verify Inference Pipeline
    # --------------------------------------------------------------------------
    print("\n[5/5] Running Inference (Debug Mode)...")

    # Run inference
    predict(load_cached_data=True, debug=True)

    # Check submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # In debug mode, we expect Config.DEBUG_SAMPLES rows
    expected_rows = Config.DEBUG_SAMPLES
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert list(df_sub.columns) == [
        "id",
        "target",
    ], f"Submission columns mismatch. Got {df_sub.columns}"

    # Check values are probabilities
    assert df_sub["target"].min() >= 0.0, "Probabilities < 0 found"
    assert df_sub["target"].max() <= 1.0, "Probabilities > 1 found"

    print(f"  - Inference complete. Submission shape: {df_sub.shape}")
    print("  - Submission format verified.")

    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    main()
