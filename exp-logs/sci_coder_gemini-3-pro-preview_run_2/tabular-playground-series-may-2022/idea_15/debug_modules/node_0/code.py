import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the library can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, calculate_auc
from library.data_loader import get_dataloaders
from library.model import (
    GatedTransformerResFunnelHybrid,
    GatedGLU,
    GatedTransformerBlock,
    ResFunnelBlock,
)
from library.train import run_training


def main():
    print("--- Starting Demo Execution ---")

    # 1. Setup Configuration for Demo
    # We override the default paths to use a clean demo directory to avoid conflicts
    demo_dir = "./working/demo_execution"
    Config.WORKING_DIR = demo_dir

    # Important: Since Config paths are defined at class level, we must update them manually
    # after changing the WORKING_DIR.
    Config.PROCESSED_DATA_PATH = os.path.join(demo_dir, "processed_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Ensure the directory exists
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(42)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Helper Functions
    print("\n[Verification] Metrics...")
    y_true_dummy = np.array([0, 1, 0, 1])
    y_pred_dummy = np.array([0.1, 0.9, 0.2, 0.8])
    auc = calculate_auc(y_true_dummy, y_pred_dummy)
    assert auc == 1.0, f"AUC calculation incorrect, expected 1.0, got {auc}"
    print("AUC function verified.")

    # 3. Verify Model Architecture
    print("\n[Verification] Model Components...")
    batch_size = 4
    cont_dim = 30
    seq_len = 10
    vocab_size = 27

    # A. GatedGLU
    glu = GatedGLU(in_features=32, hidden_features=16, out_features=32)
    dummy_input = torch.randn(batch_size, 32)
    out = glu(dummy_input)
    assert out.shape == (batch_size, 32), f"GatedGLU output shape mismatch: {out.shape}"

    # B. Transformer Block
    tf_block = GatedTransformerBlock(embed_dim=32, num_heads=4, ffn_dim=64)
    dummy_seq = torch.randn(batch_size, seq_len, 32)
    out_seq = tf_block(dummy_seq)
    assert out_seq.shape == (
        batch_size,
        seq_len,
        32,
    ), f"TransformerBlock output shape mismatch: {out_seq.shape}"

    # C. ResFunnel Block
    rf_block = ResFunnelBlock(in_features=32, out_features=16)
    out_rf = rf_block(dummy_input)
    assert out_rf.shape == (
        batch_size,
        16,
    ), f"ResFunnelBlock output shape mismatch: {out_rf.shape}"

    # D. Full Hybrid Model
    model = GatedTransformerResFunnelHybrid()
    dummy_cont = torch.randn(batch_size, cont_dim)
    dummy_cat = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(dummy_cont, dummy_cat)
    assert logits.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch: {logits.shape}"
    print("Model architecture verified.")

    # 4. Verify Data Pipeline
    print("\n[Verification] Data Pipeline...")
    # We call get_dataloaders which internally calls process_data.
    # We force reprocessing (load_cached_data=False) to verify the raw data parsing logic.
    # This creates the .npz file in the demo directory.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=2048, num_workers=2, load_cached_data=False
    )

    # Check batch structure
    batch = next(iter(train_loader))
    assert "continuous" in batch
    assert "categorical" in batch
    assert "target" in batch
    assert batch["continuous"].shape == (
        2048,
        30,
    ), f"Batch continuous shape mismatch: {batch['continuous'].shape}"
    assert batch["categorical"].shape == (
        2048,
        10,
    ), f"Batch categorical shape mismatch: {batch['categorical'].shape}"
    assert batch["target"].shape == (
        2048,
    ), f"Batch target shape mismatch: {batch['target'].shape}"
    print("Data loaders verified.")

    # 5. Execute Training Loop
    print("\n[Execution] Running Training...")
    # We run for 1 epoch to demonstrate the loop without consuming too much time.
    # We pass load_cached_data=True to reuse the data we just processed in step 4.
    run_training(
        epochs=1, batch_size=2048, lr=1e-3, weight_decay=0.01, load_cached_data=True
    )

    # 6. Verify Submission Output
    print("\n[Verification] Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Check constraints
    assert sub_df.shape == (
        100000,
        2,
    ), "Submission shape mismatch (should be 100000, 2)"
    assert list(sub_df.columns) == ["id", "target"], "Submission columns mismatch"
    assert sub_df["target"].isnull().sum() == 0, "Submission contains NaNs"

    # Check ID alignment with test metadata
    test_meta = pd.read_csv(Config.TEST_META_PATH)
    assert np.all(
        sub_df["id"].values == test_meta["id"].values
    ), "Submission IDs do not match Test Metadata"

    print("Submission verified.")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
