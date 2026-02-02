import os
import pandas as pd
import torch
import numpy as np
import shutil
from torch.optim import AdamW

# Import library components
from library.config import Config
from library.utils import set_seed, get_device
from library.vocab import load_or_build_artifacts
from library.data import get_dataloaders, get_test_dataloader
from library.model import SyntaxAwareTransformer
from library.loss import MultiTaskLoss
from library.engine import train_one_epoch, evaluate
from library.inference import generate_submission


def run_demo():
    print("--- Starting End-to-End Demo ---")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # --------------------------------------------------------------------------
    # Define a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Working directory: {DEMO_DIR}")

    # Create small subsets of metadata for speed
    # We read the first 1000 lines from the original metadata
    subset_size = 1000

    train_subset_path = os.path.join(DEMO_DIR, "train_small.csv")
    val_subset_path = os.path.join(DEMO_DIR, "val_small.csv")
    test_subset_path = os.path.join(DEMO_DIR, "test_small.csv")

    # Helper to create subset
    def create_subset(src, dst, n):
        df = pd.read_csv(src, nrows=n)
        df.to_csv(dst, index=False)
        return len(df)

    print("Creating data subsets...")
    n_train = create_subset(Config.TRAIN_METADATA_PATH, train_subset_path, subset_size)
    n_val = create_subset(Config.VAL_METADATA_PATH, val_subset_path, subset_size)
    n_test = create_subset(Config.TEST_METADATA_PATH, test_subset_path, subset_size)
    print(f"Subsets created: Train={n_train}, Val={n_val}, Test={n_test}")

    # Patch Config to use these new paths and settings
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_METADATA_PATH = train_subset_path
    Config.VAL_METADATA_PATH = val_subset_path
    Config.TEST_METADATA_PATH = test_subset_path

    # Update cache paths to be inside demo dir
    Config.TRAIN_TOKENS_PATH = os.path.join(DEMO_DIR, "train_tokens.parquet")
    Config.VAL_TOKENS_PATH = os.path.join(DEMO_DIR, "val_tokens.parquet")
    Config.TEST_TOKENS_PATH = os.path.join(DEMO_DIR, "test_tokens.parquet")
    Config.VOCAB_SAVE_PATH = os.path.join(DEMO_DIR, "vocab.npy")
    Config.POS_MAP_SAVE_PATH = os.path.join(DEMO_DIR, "pos_map.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce hyperparameters for speed
    Config.VOCAB_SIZE = 500  # Small vocab
    Config.BATCH_SIZE = 16
    Config.NUM_EPOCHS = 1
    Config.HIDDEN_DIM = 64
    Config.EMBED_DIM = 64
    Config.NUM_HEADS = 2
    Config.NUM_LAYERS = 2
    Config.DEBUG = True  # Forces data loader to respect limits if implemented

    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n--- Building Artifacts ---")
    # Force build from scratch using the small subsets
    vocab, pos_map, pos_tags = load_or_build_artifacts(load_cached_data=False)

    # Assertions for Vocab
    print(f"Vocabulary Size: {len(vocab)}")
    assert len(vocab) <= Config.VOCAB_SIZE, "Vocab size exceeds limit"
    assert vocab.lookup_token(Config.PAD_IDX) == Config.PAD_TOKEN, "PAD token mismatch"
    assert len(pos_map) == len(vocab), "POS Map length must match vocab length"

    print("\n--- Loading DataLoaders ---")
    train_loader, val_loader = get_dataloaders(batch_size=Config.BATCH_SIZE)

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    print("Sample Batch Keys:", sample_batch.keys())

    assert "input_ids" in sample_batch
    assert "loc_targets" in sample_batch
    assert (
        sample_batch["input_ids"].shape[0] == Config.BATCH_SIZE
        or sample_batch["input_ids"].shape[0] == n_train
    )
    assert sample_batch["input_ids"].dim() == 2

    print("Data Pipeline Verified.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    model = SyntaxAwareTransformer().to(device)

    # Move sample batch to device
    input_ids = sample_batch["input_ids"].to(device)
    attention_mask = sample_batch["attention_mask"].to(device)

    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)

    # Check output shapes
    batch_size, seq_len = input_ids.shape
    assert outputs["loc_logits"].shape == (
        batch_size,
        seq_len,
    ), "Loc logits shape mismatch"
    assert outputs["syntax_logits"].shape == (
        batch_size,
        seq_len,
        Config.NUM_POS_TAGS,
    ), "Syntax logits shape mismatch"
    assert outputs["word_logits"].shape == (
        batch_size,
        seq_len,
        Config.VOCAB_SIZE,
    ), "Word logits shape mismatch"

    print("Model Forward Pass Verified.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Running Training Loop (1 Epoch) ---")
    criterion = MultiTaskLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Train
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, None, criterion, device
    )
    print(f"Train Metrics: {train_metrics}")

    assert "loss" in train_metrics
    assert train_metrics["loss"] > 0, "Loss should be positive"

    # Evaluate
    print("--- Running Evaluation ---")
    val_metrics = evaluate(model, val_loader, criterion, device)
    print(f"Val Metrics: {val_metrics}")

    assert "loc_acc" in val_metrics

    # Save model (required for inference step)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Model saved.")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    # Run generation
    # Note: generate_submission in library/inference.py loads from Config.MODEL_SAVE_PATH
    generate_submission(
        batch_size=Config.BATCH_SIZE,
        model_path=Config.MODEL_SAVE_PATH,
        submission_path=Config.SUBMISSION_FILE,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission rows: {len(df_sub)}")
    print("Sample submission:\n", df_sub.head(2))

    # Validation checks on submission
    assert list(df_sub.columns) == ["id", "sentence"], "Incorrect submission columns"
    assert (
        len(df_sub) == n_test
    ), f"Submission row count {len(df_sub)} does not match test set size {n_test}"

    # Check content format (quotes)
    # We read as CSV, so pandas handles quotes. We check if the sentence is a string.
    assert isinstance(df_sub.iloc[0]["sentence"], str), "Sentence is not a string"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
