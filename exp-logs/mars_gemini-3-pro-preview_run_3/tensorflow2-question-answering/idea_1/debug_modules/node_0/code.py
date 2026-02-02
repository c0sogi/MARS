import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.models import SiameseDANRanker, ShallowCNNReader
from library.trainer import Trainer
from library.inference import NQPipeline


def verify_models():
    """
    Unit test for model architectures to ensure shapes match expectations.
    """
    print("\n[1/4] Verifying Model Architectures...")

    vocab_size = 100
    batch_size = 4
    seq_len_q = 10
    seq_len_ctx = 20
    num_candidates = 5

    # --- Test Ranker ---
    ranker = SiameseDANRanker(vocab_size=vocab_size)

    # Case 1: Single context (Training positive pair)
    q_ids = torch.randint(0, vocab_size, (batch_size, seq_len_q))
    ctx_ids_single = torch.randint(0, vocab_size, (batch_size, seq_len_ctx))
    scores_single = ranker(q_ids, ctx_ids_single)

    assert scores_single.shape == (
        batch_size,
    ), f"Ranker single context output shape mismatch. Expected {(batch_size,)}, got {scores_single.shape}"

    # Case 2: Multiple candidates (Inference/Negative sampling)
    ctx_ids_multi = torch.randint(
        0, vocab_size, (batch_size, num_candidates, seq_len_ctx)
    )
    scores_multi = ranker(q_ids, ctx_ids_multi)

    assert scores_multi.shape == (
        batch_size,
        num_candidates,
    ), f"Ranker multi-context output shape mismatch. Expected {(batch_size, num_candidates)}, got {scores_multi.shape}"

    print("  -> Ranker logic verified.")

    # --- Test Reader ---
    reader = ShallowCNNReader(vocab_size=vocab_size)

    # Input is concatenation of Q and Context
    input_len = seq_len_q + seq_len_ctx
    input_ids = torch.randint(0, vocab_size, (batch_size, input_len))

    start_logits, end_logits = reader(input_ids)

    assert start_logits.shape == (
        batch_size,
        input_len,
    ), f"Reader start logits shape mismatch. Expected {(batch_size, input_len)}, got {start_logits.shape}"
    assert end_logits.shape == (
        batch_size,
        input_len,
    ), f"Reader end logits shape mismatch. Expected {(batch_size, input_len)}, got {end_logits.shape}"

    print("  -> Reader logic verified.")


def run_training_demo():
    """
    Demonstrates the training pipeline using a small subset of data.
    """
    print("\n[2/4] Running Training Demonstration...")

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.MAX_TRAIN_SAMPLES = 50  # Limit vocab build size

    # Initialize Trainer with a small debug sample size
    # This triggers get_dataloaders -> build_vocab -> preprocess_annotations
    debug_size = 50
    trainer = Trainer(debug_sample_size=debug_size)

    # Run training
    trainer.run_training()

    # Verify checkpoints exist
    ranker_path = os.path.join(Config.CACHE_DIR, "ranker_best.pth")
    reader_path = os.path.join(Config.CACHE_DIR, "reader_best.pth")

    assert os.path.exists(ranker_path), "Ranker checkpoint was not created."
    assert os.path.exists(reader_path), "Reader checkpoint was not created."

    print(f"  -> Training completed. Checkpoints saved to {Config.CACHE_DIR}")


def run_inference_demo():
    """
    Demonstrates the inference pipeline.
    """
    print("\n[3/4] Running Inference Demonstration...")

    # Initialize Pipeline with small subset of test data
    debug_size = 20
    pipeline = NQPipeline(debug_sample_size=debug_size)

    # Run prediction loop
    pipeline.generate_predictions()

    # Verify submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    print(f"  -> Inference completed. Submission saved to {Config.SUBMISSION_PATH}")


def validate_submission():
    """
    Validates the generated submission file format.
    """
    print("\n[4/4] Validating Submission File...")

    df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["example_id", "PredictionString"]
    assert (
        list(df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

    # Check content
    # We expect 2 rows per example (long and short)
    # If we ran inference on ~20 examples, we expect roughly 40 rows
    print(f"  -> Submission contains {len(df)} rows.")

    if len(df) > 0:
        sample_row = df.iloc[0]
        print(f"  -> Sample row: {sample_row.to_dict()}")

        # Verify ID format (should end in _long or _short)
        ex_id = sample_row["example_id"]
        assert ex_id.endswith("_long") or ex_id.endswith(
            "_short"
        ), f"Invalid example_id format: {ex_id}"

    print("  -> Submission format verified.")


if __name__ == "__main__":
    # Set seeds for reproducibility (Config.setup() does this, but doing it here ensures main script consistency)
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== Natural Questions Pipeline Demonstration ===")

    try:
        # 1. Verify Model Logic
        verify_models()

        # 2. Run Training Pipeline
        run_training_demo()

        # 3. Run Inference Pipeline
        run_inference_demo()

        # 4. Validate Output
        validate_submission()

        print("\n=== Demonstration Completed Successfully ===")

    except AssertionError as e:
        print(f"\n!!! Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
