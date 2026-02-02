import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Ensure the current working directory is in the path for module imports
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import BalancedProcessCompressHybrid, SequenceEncoder
from library.train_eval import run_training


def main():
    # 1. Setup
    print(">>> Setting up environment...")
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    # Optimize Config for Speed in this Demo
    print(">>> Adjusting configuration for demo speed...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2048  # Larger batch size for faster iteration on tabular data

    # 2. Data Loading Verification
    print(">>> Testing Data Loading...")
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Reduced workers for simple demo to avoid overhead
        load_cached_data=True,
    )

    # Fetch one batch to verify shapes
    x_num_batch, x_cat_batch, y_batch = next(iter(train_loader))

    print(f"    Vocab Size: {vocab_size}")
    print(
        f"    Batch Shapes -> Num: {x_num_batch.shape}, Cat: {x_cat_batch.shape}, Target: {y_batch.shape}"
    )

    # Assertions for Data Integrity
    # x_num should be (Batch, 30) based on f_00 to f_30 excluding f_27
    assert (
        x_num_batch.shape[1] == 30
    ), f"Expected 30 numerical features, got {x_num_batch.shape[1]}"
    # x_cat should be (Batch, 10) based on f_27 string length
    assert (
        x_cat_batch.shape[1] == 10
    ), f"Expected sequence length 10, got {x_cat_batch.shape[1]}"
    assert y_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    print("    [OK] Data Loader shapes verified.")

    # 3. Component Unit Testing
    print(">>> Testing Model Components...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. Test Sequence Encoder
    seq_len = 10
    embed_dim = 32
    encoder = SequenceEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_len=seq_len,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    ).to(device)

    dummy_cat = torch.randint(0, vocab_size, (32, seq_len)).to(device)
    encoded = encoder(dummy_cat)

    # Expected output shape: (Batch, Seq_Len * Embed_Dim)
    expected_dim = seq_len * embed_dim
    assert encoded.shape == (
        32,
        expected_dim,
    ), f"Encoder output mismatch. Expected (32, {expected_dim}), got {encoded.shape}"
    print("    [OK] SequenceEncoder forward pass successful.")

    # B. Test Full Model
    model = BalancedProcessCompressHybrid(
        num_continuous=30,
        cat_seq_len=10,
        vocab_size=vocab_size,
        embed_dim=16,
        transformer_layers=1,
        transformer_heads=2,
        backbone_stages=[64, 32],  # Smaller backbone for test
        dropout_transformer=0.1,
        dropout_backbone=0.1,
    ).to(device)

    dummy_num = torch.randn(32, 30).to(device)
    output = model(dummy_num, dummy_cat)

    assert output.shape == (
        32,
        1,
    ), f"Model output shape mismatch. Expected (32, 1), got {output.shape}"
    assert not torch.isnan(output).any(), "Model produced NaN values"
    print("    [OK] Full Model forward pass successful.")

    # 4. Full Pipeline Execution
    print(">>> Executing Full Training Pipeline (1 Epoch)...")

    # We use the run_training function from library.train_eval
    # This handles training loop, validation, and submission generation
    run_training(epochs=1, batch_size=Config.BATCH_SIZE, debug=True)

    # 5. Artifact Validation
    print(">>> Verifying Output Artifacts...")

    # Check Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {df_sub.shape}")

    # Check row count (Test set is 100,000 rows)
    assert len(df_sub) == 100000, f"Expected 100,000 predictions, found {len(df_sub)}"

    # Check columns
    assert list(df_sub.columns) == ["id", "target"], "Submission columns mismatch"

    # Check Model Checkpoint
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    print("    [OK] Artifacts verified successfully.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
