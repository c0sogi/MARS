import os
import sys
import pandas as pd
import torch
import shutil

# Import provided library modules
from library.config import Config, set_seed
from library.data_loader import get_dataloaders, Vocabulary
from library.model import BiLSTMTagger
from library.trainer import Trainer
from library.inference import generate_submission


def run_demo():
    # ---------------------------------------------------------
    # 1. Setup and Configuration Override
    # ---------------------------------------------------------
    print("[Demo] Setting up configuration for fast execution...")

    # Override Config parameters to ensure the demo runs quickly
    Config.MAX_TRAIN_SAMPLES = 1000  # Limit training to 1000 sentences
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 64  # Smaller batch size
    Config.EMBEDDING_DIM = 64  # Reduced embedding dimension
    Config.HIDDEN_DIM = 128  # Reduced hidden dimension
    Config.NUM_LAYERS = 1  # Single LSTM layer
    Config.BIDIRECTIONAL = True  # Keep bidirectional

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"[Demo] Device: {Config.DEVICE}")
    print(f"[Demo] Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("\n[Demo] Preparing Data (Loading, Grouping, Vocab Building)...")

    # Call get_dataloaders.
    # load_cached_data=False ensures we process the raw metadata files.
    # debug=True triggers the subsampling logic based on Config.MAX_TRAIN_SAMPLES.
    train_loader, val_loader, test_loader, vocab, kb = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Validate Data Loading
    print(f"[Demo] Train batches: {len(train_loader)}")
    print(f"[Demo] Val batches: {len(val_loader)}")
    print(f"[Demo] Vocab size: {len(vocab.token2id)}")
    print(f"[Demo] Knowledge Base entries: {len(kb)}")

    assert len(train_loader) > 0, "Train loader is empty"
    assert (
        len(vocab.token2id) > 2
    ), "Vocabulary should contain more than just special tokens"
    assert len(kb) > 0, "Knowledge base is empty"

    # Validate Batch Structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "labels" in sample_batch
    assert (
        sample_batch["input_ids"].shape[0] == Config.BATCH_SIZE
        or sample_batch["input_ids"].shape[0]
        == Config.MAX_TRAIN_SAMPLES % Config.BATCH_SIZE
    )
    print(f"[Demo] Sample Batch Input Shape: {sample_batch['input_ids'].shape}")

    # ---------------------------------------------------------
    # 3. Test Data Truncation (Optimization for Demo)
    # ---------------------------------------------------------
    # The full test set has ~1M sentences. To demonstrate inference quickly,
    # we modify the cached test parquet file to only contain a small subset.
    print("\n[Demo] Truncating test data cache for rapid inference demonstration...")
    test_grouped_path = os.path.join(Config.WORKING_DIR, "test_grouped.parquet")

    if os.path.exists(test_grouped_path):
        df_test_full = pd.read_parquet(test_grouped_path)
        original_len = len(df_test_full)

        # Keep only top 100 sentences for the inference demo
        df_test_small = df_test_full.head(100)
        df_test_small.to_parquet(test_grouped_path)

        print(
            f"[Demo] Test set truncated from {original_len} to {len(df_test_small)} sentences."
        )
    else:
        raise FileNotFoundError(f"Test cache not found at {test_grouped_path}")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[Demo] Initializing BiLSTMTagger...")
    model = BiLSTMTagger(
        vocab_size=len(vocab.token2id),
        num_classes=len(vocab.class2id),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        bidirectional=Config.BIDIRECTIONAL,
    )
    model.to(Config.DEVICE)
    print("[Demo] Model initialized successfully.")

    # ---------------------------------------------------------
    # 5. Training
    # ---------------------------------------------------------
    print("\n[Demo] Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, vocab)
    trainer.fit()

    # Validate Model Checkpoint
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint file was not created."
    print(f"[Demo] Model saved to {Config.MODEL_CHECKPOINT_PATH}")

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[Demo] Generating Submission...")
    # This will load the trained model and the truncated test data
    generate_submission(debug=True)

    # Validate Submission File
    submission_path = Config.SUBMISSION_FILE_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    df_submission = pd.read_csv(submission_path)
    print(f"[Demo] Submission loaded. Rows: {len(df_submission)}")
    print(df_submission.head())

    # Check format
    assert "id" in df_submission.columns
    assert "after" in df_submission.columns
    assert len(df_submission) > 0, "Submission file is empty."

    print("\n[Demo] All steps completed successfully.")


if __name__ == "__main__":
    run_demo()
