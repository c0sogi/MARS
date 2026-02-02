import os
import sys
import shutil
import warnings
import logging
import torch
import pandas as pd
import numpy as np
from functools import partial

# ==========================================
# 0. Environment & Logging Setup
# ==========================================

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Patch tqdm to disable progress bars globally
import tqdm.auto


def noop_tqdm(*args, **kwargs):
    kwargs["disable"] = True
    return tqdm.auto.tqdm(*args, **kwargs)


# We need to patch the class itself or the import in the library.
# Since the library imports `from tqdm.auto import tqdm`, we patch it here
# before importing the library modules.
tqdm.auto.tqdm = partial(tqdm.auto.tqdm, disable=True)

# Import Library Modules
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.data import EmbeddingManager, NotebookDataset, collate_fn
from library.model import SemanticAnchorClassifier
from library.train import run_training
from library.inference import generate_submission


def main():
    print("=== AI4Code Solution Demonstration ===")

    # ==========================================
    # 1. Configuration
    # ==========================================
    print("\n[1] Initializing Configuration...")

    # Define a specific working directory for this demo
    demo_working_dir = "./working/demo_run"

    # Clean up previous run if exists
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)

    # Initialize Config with Debug settings for speed
    config = Config(
        debug=True,
        debug_sample_size=20,  # Process only 20 notebooks for speed
        working_dir=demo_working_dir,
        train_batch_size=4,
        val_batch_size=4,
        epochs=1,  # Run only 1 epoch
        num_workers=0,  # Disable multiprocessing for simple demo
        early_stopping_patience=1,
    )

    # Ensure reproducibility
    set_seed(config.seed)
    print(f"    Working Directory: {config.working_dir}")
    print(f"    Debug Mode: {config.debug}")
    print(f"    Device: {config.device}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")

    manager = EmbeddingManager(config)

    # A. Process Training Data
    print("    Processing training data...")
    df_train = manager.process_data("train")

    # Assertions for DataFrame structure
    assert not df_train.empty, "Training DataFrame should not be empty."
    required_cols = [
        "id",
        "code_embeddings",
        "markdown_embeddings",
        "labels",
        "code_ids",
        "markdown_ids",
    ]
    for col in required_cols:
        assert col in df_train.columns, f"Missing column {col} in training data."

    # B. Process Validation Data
    print("    Processing validation data...")
    df_val = manager.process_data("val")
    assert not df_val.empty, "Validation DataFrame should not be empty."

    # C. Verify Dataset and Collate Function
    print("    Verifying Dataset and Batch Collation...")
    train_dataset = NotebookDataset(df_train)

    # Fetch a few samples to create a batch
    batch_samples = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
    batch = collate_fn(batch_samples)

    # Verify Batch Structure
    assert "code_embeddings" in batch
    assert "markdown_embeddings" in batch
    assert "labels" in batch
    assert "ids" in batch

    # Verify Tensor Shapes
    # code_embeddings: (Batch, Max_Code_Len, Input_Dim)
    c_emb = batch["code_embeddings"]
    m_emb = batch["markdown_embeddings"]
    labels = batch["labels"]

    assert c_emb.dim() == 3, f"Code embeddings should be 3D, got {c_emb.shape}"
    assert m_emb.dim() == 3, f"Markdown embeddings should be 3D, got {m_emb.shape}"
    assert labels.dim() == 2, f"Labels should be 2D, got {labels.shape}"
    assert (
        c_emb.size(2) == config.input_dim
    ), f"Embedding dim mismatch. Expected {config.input_dim}, got {c_emb.size(2)}"

    print("    -> Data pipeline verified successfully.")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("\n[3] Verifying Model Logic...")

    model = SemanticAnchorClassifier(config)
    model.to(config.device)
    model.eval()

    # Move batch to device
    c_emb = c_emb.to(config.device)
    m_emb = m_emb.to(config.device)

    # Perform Forward Pass
    with torch.no_grad():
        # Handle case where sampled notebooks might have 0 markdown cells (unlikely but possible)
        if m_emb.size(1) > 0:
            logits = model(m_emb, c_emb)

            # Expected Shape: (Batch, Num_MD, Num_Code + 1)
            # Num_Code + 1 accounts for the End-of-Notebook token
            expected_classes = c_emb.size(1) + 1
            expected_shape = (c_emb.size(0), m_emb.size(1), expected_classes)

            assert (
                logits.shape == expected_shape
            ), f"Logits shape mismatch. Expected {expected_shape}, got {logits.shape}"
            print(f"    -> Forward pass successful. Logits shape: {logits.shape}")
        else:
            print(
                "    -> Skipped forward pass check (no markdown cells in random batch)."
            )

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n[4] Executing Training Loop...")

    # Run the full training routine provided in library
    run_training(config)

    # Verify Model Checkpoint
    assert os.path.exists(
        config.model_save_path
    ), f"Model checkpoint not found at {config.model_save_path}"
    print(f"    -> Training complete. Model saved to {config.model_save_path}")

    # ==========================================
    # 5. Inference Execution
    # ==========================================
    print("\n[5] Executing Inference...")

    # Run the inference routine provided in library
    generate_submission(config)

    # Verify Submission File
    assert os.path.exists(
        config.submission_path
    ), f"Submission file not found at {config.submission_path}"

    df_sub = pd.read_csv(config.submission_path)
    assert not df_sub.empty, "Submission file is empty."
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission file missing required columns."

    # Check if we predicted for the debug sample size
    # Note: process_data uses debug_sample_size for test set too when debug=True
    assert (
        len(df_sub) == config.debug_sample_size
    ), f"Expected {config.debug_sample_size} predictions, got {len(df_sub)}"

    print(f"    -> Inference complete. Submission generated with {len(df_sub)} rows.")

    # ==========================================
    # 6. Metric Utility Verification
    # ==========================================
    print("\n[6] Verifying Metric Calculation (Kendall Tau)...")

    # Test Case:
    # Ground Truth: [A, B, C]
    # Prediction:   [A, C, B]
    #
    # Pairs in GT: (A,B), (A,C), (B,C) -> All ordered correctly
    # Pairs in Pred: (A,C), (A,B), (C,B)
    #
    # Inversions relative to GT:
    # A comes before B (Correct)
    # A comes before C (Correct)
    # C comes before B (Incorrect -> 1 Inversion)
    #
    # Total Swaps (S) = 1
    # n = 3
    # Total Possible = n(n-1) = 3*2 = 6
    # Kendall Tau = 1 - 4 * (S / Total Possible)
    # K = 1 - 4 * (1 / 6) = 1 - 0.666... = 0.333...

    gt = {"nb_1": ["A", "B", "C"]}
    pred = {"nb_1": ["A", "C", "B"]}

    score = compute_kendall_tau(gt, pred)
    expected_score = 1 - 4 * (1 / 6)

    assert (
        abs(score - expected_score) < 1e-6
    ), f"Metric calculation failed. Expected {expected_score}, got {score}"

    # Test Perfect Match
    pred_perfect = {"nb_1": ["A", "B", "C"]}
    score_perfect = compute_kendall_tau(gt, pred_perfect)
    assert abs(score_perfect - 1.0) < 1e-6, "Metric should be 1.0 for perfect match."

    print("    -> Metric logic verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
