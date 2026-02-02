import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.data import (
    get_dataloaders,
    get_tokenizer,
    process_text_data,
    get_target_columns,
)
from library.model import QuestModel
from library.engine import run_training


def main():
    print("=== Starting Demonstration of Google QUEST Q&A Labeling Solution ===\n")

    # ==========================================
    # 1. Configuration and Setup
    # ==========================================
    print("[1] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.debug = True  # Uses small subset (100 samples)
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Set up a specific working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.working_dir = demo_dir
    Config.model_save_path = os.path.join(demo_dir, "best_model.pth")
    Config.train_cache_path = os.path.join(demo_dir, "train_cached.parquet")
    Config.val_cache_path = os.path.join(demo_dir, "val_cached.parquet")
    Config.test_cache_path = os.path.join(demo_dir, "test_cached.parquet")
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    # Set seed for reproducibility
    seed_everything(Config.seed)
    print("    Configuration updated. Debug mode: ON. Epochs: 1.")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")

    # Initialize Tokenizer
    tokenizer = get_tokenizer()
    print(f"    Tokenizer initialized: {tokenizer.__class__.__name__}")

    # Create a dummy dataframe to test processing logic
    dummy_data = {
        Config.qa_id_col: [1, 2],
        Config.question_title_col: ["How to use Python?", "What is PyTorch?"],
        Config.question_body_col: ["I am learning coding.", "It is a library."],
        Config.answer_col: ["Python is great.", "PyTorch is a framework."],
    }
    # Add dummy targets
    target_cols = get_target_columns()
    for col in target_cols:
        dummy_data[col] = [0.5, 0.5]

    df_dummy = pd.DataFrame(dummy_data)

    # Test process_text_data
    processed_samples = process_text_data(
        df_dummy, tokenizer, target_cols=target_cols, is_test=False
    )

    assert len(processed_samples) == 2, "Processed samples count mismatch"
    sample = processed_samples[0]

    # Verify keys
    required_keys = [
        "qa_id",
        "input_ids",
        "attention_mask",
        "question_mask",
        "answer_mask",
        "labels",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in processed sample: {key}"

    # Verify Mask Logic
    # Question mask and Answer mask should be mutually exclusive (1s should not overlap)
    q_mask = np.array(sample["question_mask"])
    a_mask = np.array(sample["answer_mask"])
    overlap = (q_mask * a_mask).sum()
    assert (
        overlap == 0
    ), "Question and Answer masks should not overlap (token cannot be both)"

    # Verify at least some tokens are masked
    assert q_mask.sum() > 0, "Question mask is empty"
    assert a_mask.sum() > 0, "Answer mask is empty"

    print("    Data processing logic (tokenization & masking) verified.")

    # Load actual dataloaders (will use cached/debug data)
    print("    Loading DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Check one batch from train_loader
    batch = next(iter(train_loader))
    print(f"    Batch keys: {list(batch.keys())}")
    print(f"    Input IDs shape: {batch['input_ids'].shape}")
    print(f"    Labels shape: {batch['labels'].shape}")

    assert batch["input_ids"].shape[0] == Config.train_batch_size
    assert batch["labels"].shape[1] == 30
    assert batch["question_mask"].shape == batch["input_ids"].shape

    print("    DataLoaders verified.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = Config.device
    model = QuestModel()
    model.to(device)
    model.eval()

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    question_mask = batch["question_mask"].to(device)
    answer_mask = batch["answer_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_mask=question_mask,
            answer_mask=answer_mask,
        )

    print(f"    Logits shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.train_batch_size,
        30,
    ), f"Expected output shape ({Config.train_batch_size}, 30), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("    Model forward pass successful.")

    # ==========================================
    # 4. Metric Verification
    # ==========================================
    print("\n[4] Verifying Metric (Spearman's Correlation)...")

    # Case 1: Perfect correlation
    preds_perfect = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    targets_perfect = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    score_perfect = compute_spearmanr(preds_perfect, targets_perfect)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected 1.0 for perfect correlation, got {score_perfect}"

    # Case 2: Negative correlation
    preds_neg = np.array([[0.1], [0.5], [0.9]])
    targets_neg = np.array([[0.9], [0.5], [0.1]])
    score_neg = compute_spearmanr(preds_neg, targets_neg)
    assert np.isclose(
        score_neg, -1.0
    ), f"Expected -1.0 for negative correlation, got {score_neg}"

    print(
        f"    Metric verification passed. Perfect score: {score_perfect}, Inverse score: {score_neg}"
    )

    # ==========================================
    # 5. Full Training Loop Execution
    # ==========================================
    print("\n[5] Executing Training Loop (Engine)...")

    # We use the engine's run_training function
    # This will train for 1 epoch on the debug subset (100 samples)
    # It will save the model and generate a submission file

    try:
        run_training(train_loader, val_loader, test_loader)
        print("    Training execution completed without errors.")
    except Exception as e:
        print(f"    Training execution failed: {e}")
        raise e

    # ==========================================
    # 6. Output Verification
    # ==========================================
    print("\n[6] Verifying Submission Output...")

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    sub_df = pd.read_csv(Config.submission_path)
    print(f"    Submission shape: {sub_df.shape}")
    print(f"    Submission columns: {sub_df.columns.tolist()[:3]} ...")

    # Assertions
    # In debug mode, test loader loads 100 samples.
    expected_rows = 100 if Config.debug else 608
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"
    assert Config.qa_id_col in sub_df.columns, "qa_id column missing"
    assert len(sub_df.columns) == 31, "Expected 31 columns (1 ID + 30 targets)"

    # Check value range
    target_vals = sub_df.drop(columns=[Config.qa_id_col]).values
    assert (target_vals >= 0).all() and (
        target_vals <= 1
    ).all(), "Predictions out of range [0, 1]"

    print("    Submission file verified successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
