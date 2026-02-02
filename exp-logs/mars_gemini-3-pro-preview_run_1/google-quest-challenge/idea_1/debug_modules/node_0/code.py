import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.data_loader import get_dataloaders, Tokenizer
from library.model import DualBranchDAN
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100  # Use only 100 samples for this demo
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size

    # Ensure directories exist
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated: DEBUG=True, EPOCHS=2, DEBUG_SIZE=100")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[2] Verifying Data Loading and Processing...")

    # We force `load_cached_data=False` to demonstrate raw processing logic at least once.
    # Note: In a real run, we would prefer True to save time.
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False, debug=Config.DEBUG
    )

    # Verify Tokenizer
    assert isinstance(tokenizer, Tokenizer)
    print(f"    Tokenizer vocabulary size: {len(tokenizer.word2idx)}")

    # Verify Train Loader Batch
    try:
        batch = next(iter(train_loader))
        q_seq, a_seq, targets = batch

        print(
            f"    Batch Shapes -> Q: {q_seq.shape}, A: {a_seq.shape}, Targets: {targets.shape}"
        )

        # Assertions
        assert q_seq.shape == (
            Config.BATCH_SIZE,
            Config.MAX_LEN_Q,
        ), "Question sequence shape mismatch"
        assert a_seq.shape == (
            Config.BATCH_SIZE,
            Config.MAX_LEN_A,
        ), "Answer sequence shape mismatch"
        assert targets.shape == (
            Config.BATCH_SIZE,
            len(Config.TARGET_COLS),
        ), "Target shape mismatch"
        assert q_seq.dtype == torch.long, "Question tensor should be Long type"
        assert targets.dtype == torch.float32, "Targets should be Float32"

        print("    Data Loading checks passed.")

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    model = DualBranchDAN(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        num_targets=len(Config.TARGET_COLS),
    )

    # Move to CPU for this quick check (or GPU if available, but CPU is safer for simple logic check)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Create dummy input based on config
    dummy_q = torch.zeros((4, Config.MAX_LEN_Q), dtype=torch.long).to(device)
    dummy_a = torch.zeros((4, Config.MAX_LEN_A), dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(dummy_q, dummy_a)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (4, len(Config.TARGET_COLS)), "Model output shape mismatch"
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Model predictions must be in [0, 1] (Sigmoid)"
    print("    Model architecture checks passed.")

    # ==========================================
    # 4. Metric Calculation Verification
    # ==========================================
    print("\n[4] Verifying Metric Calculation (Spearman)...")

    # Case A: Perfect Correlation
    y_true = torch.rand((20, 30))
    y_pred_perfect = y_true.clone()
    score_perfect = compute_spearman_metric(y_pred_perfect, y_true)
    print(f"    Perfect Correlation Score: {score_perfect:.4f}")
    assert np.isclose(score_perfect, 1.0), "Metric should be 1.0 for identical inputs"

    # Case B: Random Data (Just ensure it runs without error and returns valid range)
    y_pred_rand = torch.rand((20, 30))
    score_rand = compute_spearman_metric(y_pred_rand, y_true)
    print(f"    Random Correlation Score: {score_rand:.4f}")
    assert -1.0 <= score_rand <= 1.0, "Spearman correlation must be between -1 and 1"

    print("    Metric calculation checks passed.")

    # ==========================================
    # 5. Full Training Loop (Trainer)
    # ==========================================
    print("\n[5] Running Trainer (Fit Loop)...")

    trainer = Trainer()

    # Run training
    # This will use the Config.DEBUG settings (2 epochs, 100 samples)
    trainer.fit()

    # Verify Best Model Saved
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Success: Best model found at {best_model_path}")
    else:
        # It's possible validation didn't improve if random init was lucky,
        # but usually it saves at least once. If not, we check if logic allowed it.
        # For this demo, we just warn, but strictly we expect a save if patience > 0
        print(
            "    Notice: No best model file found (might be due to short epochs/no improvement)."
        )

    # ==========================================
    # 6. Prediction & Submission
    # ==========================================
    print("\n[6] Running Prediction on Test Set...")

    trainer.predict()

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission DataFrame Shape: {sub_df.shape}")
    print(f"    First few columns: {list(sub_df.columns[:5])}")

    # Assertions
    # We used DEBUG_SIZE=100. The test loader respects this.
    # So we expect 100 rows in submission.
    expected_rows = Config.DEBUG_SIZE
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, found {len(sub_df)}"

    # Check columns
    expected_cols = ["qa_id"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements"

    # Check value ranges
    numeric_cols = sub_df.columns[1:]
    assert (sub_df[numeric_cols] >= 0).all().all() and (
        sub_df[numeric_cols] <= 1
    ).all().all(), "Submission values must be probabilities [0, 1]"

    print("    Submission file verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
