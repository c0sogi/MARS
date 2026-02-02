import os
import torch
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.symbolic_model import SymbolicMemory
from library.trainer import run_training
from library.inference import HybridNormalizer


def main():
    print("=== Starting Demonstration of Text Normalization Library ===")

    # 1. Setup and Configuration
    # We override the configuration to ensure the demo runs quickly and uses minimal resources.
    print("\n[Step 1] Configuring environment...")
    set_seed(42)

    # Override Config parameters for a fast demonstration
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 1
    Config.EMBED_DIM = 32
    Config.HIDDEN_DIM = 64
    Config.NUM_LAYERS = 1
    Config.PATIENCE = 1

    print(
        f"Configuration overridden: Batch Size={Config.BATCH_SIZE}, Epochs={Config.NUM_EPOCHS}"
    )
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Processing Demonstration
    # We verify that the data loader produces batches with the correct shape and content.
    print("\n[Step 2] Demonstrating Data Processing Pipeline...")

    # We use a small debug_sample_size to avoid loading the entire dataset into the DataLoader
    # load_cached_data=False forces the processing logic to run (context generation, filtering)
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False, debug_sample_size=500
    )

    # Verify Vocabulary
    vocab_size = len(tokenizer)
    print(f"Vocabulary Size: {vocab_size}")
    assert vocab_size > 0, "Vocabulary should not be empty."

    # Verify Batch Structure
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]

    print(f"Sample Batch Keys: {list(batch.keys())}")
    print(f"Input IDs Shape: {input_ids.shape}")
    print(f"Target IDs Shape: {target_ids.shape}")

    # Assertions
    assert (
        input_ids.shape[0] == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {input_ids.shape[0]}"
    assert target_ids.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"
    assert "attention_mask" in batch, "Attention mask missing from batch"

    # 3. Symbolic Model Demonstration
    # The SymbolicMemory builds N-gram stats from the full training set.
    # This ensures high precision for common patterns.
    print("\n[Step 3] Demonstrating Symbolic Memory (N-gram Stats)...")

    symbolic_mem = SymbolicMemory()

    # fit() will compute stats from 'metadata/train.parquet' and cache them.
    # This might take a minute or two as it processes 7M rows, but it's essential for the hybrid model.
    symbolic_mem.fit(load_cached_data=True)

    # Verify that stats are populated
    print(f"Trigrams loaded: {len(symbolic_mem.trigrams)}")
    print(f"Unigrams loaded: {len(symbolic_mem.unigrams)}")

    # Basic query verification (assuming common words exist in the training data)
    # We query for a token that likely exists and shouldn't change (e.g., 'the')
    # Note: The symbolic memory is built on the full dataset, so 'the' -> 'the' should be a strong unigram.
    test_token = "the"
    result = symbolic_mem.query("", test_token, "")
    print(f"Symbolic Query for '{test_token}': {result}")

    # 4. Neural Model Training Demonstration
    # We train the Seq2Seq model on a small subset of data to demonstrate the training loop.
    print("\n[Step 4] Demonstrating Neural Model Training...")

    # We use a debug_sample_size to make training instant.
    # load_cached_data=True reuses the processing done in Step 2 (if applicable) or loads from disk.
    run_training(debug_sample_size=200, load_cached_data=True)

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")

    # 5. Inference Demonstration
    # We run the full hybrid inference pipeline on a subset of the test data.
    print("\n[Step 5] Demonstrating Hybrid Inference...")

    # Initialize the normalizer (loads tokenizer, symbolic memory, and neural model)
    normalizer = HybridNormalizer(load_cached_data=True)

    # Predict on a small subset of the test set
    # This generates the submission file.
    normalizer.predict_dataset(load_cached_data=True, debug_sample_size=100)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Shape: {df_submission.shape}")
    print("First 5 rows of submission:")
    print(df_submission.head())

    # Validate format
    expected_columns = ["id", "after"]
    assert (
        list(df_submission.columns) == expected_columns
    ), f"Expected columns {expected_columns}, got {list(df_submission.columns)}"
    assert (
        len(df_submission) == 100
    ), "Submission row count does not match debug sample size."

    # Check for empty predictions
    null_preds = df_submission["after"].isnull().sum()
    assert null_preds == 0, f"Found {null_preds} null predictions in submission."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
