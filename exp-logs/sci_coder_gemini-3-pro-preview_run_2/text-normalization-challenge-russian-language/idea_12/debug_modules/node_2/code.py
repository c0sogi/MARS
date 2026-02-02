import os
import shutil
import pandas as pd
import torch
import numpy as np

# Import components from the provided library
from library.config import (
    ModelConfig,
    setup_environment,
    TRAIN_META_PATH,
    VAL_META_PATH,
    WORKING_DIR,
    CHECKPOINT_DIR,
    TOKENIZER_DIR,
)
from library.text_utils import build_char_vocab, train_bpe_tokenizer
from library.hfbb_engine import HFBBModel
from library.dataset_factory import get_dataloaders
from library.trainer import train_model
from library.inference_engine import HybridNormalizer
from library.transformer_model import CharToBPESeq2Seq


def main():
    print("=== Starting Text Normalization Pipeline Demo ===")

    # 1. Configuration
    # Optimize for speed: small subset, minimal epochs, small vocab
    config = ModelConfig(debug=True, seed=42)
    config.subset_size = 500  # Use only 500 sentences
    config.num_epochs = 1
    config.batch_size = 16
    config.bpe_vocab_size = 500  # Small vocab for the small subset
    config.char_vocab_size = 100
    config.context_window = 2

    # Clean working directory to ensure we demonstrate full creation process
    if os.path.exists(WORKING_DIR):
        print(f"Cleaning working directory: {WORKING_DIR}")
        shutil.rmtree(WORKING_DIR)

    # Setup environment (creates directories, sets seeds)
    setup_environment(seed=42)

    # 2. Data Preparation (Subset)
    print("\n>>> Loading and subsetting data...")
    # We manually slice the dataframe to pass to components that expect a DF
    df_train_full = pd.read_csv(TRAIN_META_PATH)
    train_sents = df_train_full["sentence_id"].unique()[: config.subset_size]
    df_train_subset = df_train_full[
        df_train_full["sentence_id"].isin(train_sents)
    ].copy()

    print(f"Full Train Shape: {df_train_full.shape}")
    print(f"Subset Train Shape: {df_train_subset.shape}")

    # 3. Build Tokenizers
    print("\n>>> Building Tokenizers...")
    # Force rebuild with load_cached_data=False
    char_tokenizer = build_char_vocab(
        df_train_subset, vocab_size=config.char_vocab_size, load_cached_data=False
    )

    bpe_tokenizer = train_bpe_tokenizer(
        df_train_subset, vocab_size=config.bpe_vocab_size, load_cached_data=False
    )

    # Verification
    print(f"Char Vocab Size: {char_tokenizer.vocab_size}")
    print(f"BPE Vocab Size: {len(bpe_tokenizer)}")

    if char_tokenizer.vocab_size == 0:
        raise AssertionError("Character tokenizer vocabulary is empty!")
    if len(bpe_tokenizer) != config.bpe_vocab_size:
        # Note: SentencePiece might produce slightly fewer tokens if data is very small,
        # but with 500 sentences and vocab 500 it should match or be close.
        # We'll just warn if it's drastically different, or assert it exists.
        assert len(bpe_tokenizer) > 0, "BPE tokenizer is empty!"

    # 4. Build HFBB Model (Tier 1)
    print("\n>>> Building HFBB Model (Tier 1)...")
    hfbb = HFBBModel(config)
    hfbb.build(df_train_subset, load_cached_data=False)

    # Verification: Check if internal dictionaries are populated
    print(f"HFBB Unigram Dict Size: {len(hfbb.unigram_dict)}")
    if len(hfbb.unigram_dict) == 0:
        raise AssertionError("HFBB Model failed to populate unigram dictionary.")

    # Test Query on a token known to exist
    sample_token = df_train_subset["before"].iloc[0]
    # We pass empty context just to check unigram fallback
    prediction = hfbb.query(sample_token, "", "")
    print(f"HFBB Query Test: '{sample_token}' -> '{prediction}'")

    # 5. Train Transformer Model (Tier 2)
    print("\n>>> Training Transformer Model (Tier 2)...")

    # Get DataLoaders (This handles context formatting and class balancing)
    # Note: get_dataloaders uses config.subset_size internally when debug=True
    train_loader, val_loader = get_dataloaders(
        config, char_tokenizer, bpe_tokenizer, load_cached_data=False
    )

    # Train the model
    model = train_model(
        config,
        train_loader,
        val_loader,
        char_vocab_size=char_tokenizer.vocab_size,
        bpe_vocab_size=len(bpe_tokenizer),
    )

    # Verification: Check for checkpoint
    expected_checkpoint = os.path.join(CHECKPOINT_DIR, "transformer_best.pth")
    if not os.path.exists(expected_checkpoint):
        raise AssertionError(f"Checkpoint not found at {expected_checkpoint}")
    print(f"Transformer training complete. Checkpoint saved.")

    # 6. Run Inference
    print("\n>>> Running Inference Pipeline...")

    # Prepare a dummy test set from validation data
    df_val_full = pd.read_csv(VAL_META_PATH)
    val_sents = df_val_full["sentence_id"].unique()[:10]  # Take 10 sentences
    df_test_dummy = df_val_full[df_val_full["sentence_id"].isin(val_sents)].copy()

    # Ensure we have required columns for test (id, sentence_id, token_id, before)
    # The 'id' column is usually "sentence_id_token_id"
    df_test_dummy["id"] = (
        df_test_dummy["sentence_id"].astype(str)
        + "_"
        + df_test_dummy["token_id"].astype(str)
    )
    df_test_dummy = df_test_dummy[["id", "sentence_id", "token_id", "before"]]

    print(f"Test Subset Shape: {df_test_dummy.shape}")

    # Initialize Normalizer
    normalizer = HybridNormalizer(config)

    # Load resources (Tokenizers, HFBB, Transformer)
    # Note: This will load the files we just created/trained
    normalizer.load_resources()

    # Predict
    submission_df = normalizer.predict(df_test_dummy)

    # Verification
    print("\nSubmission Head:")
    print(submission_df.head())

    if len(submission_df) != len(df_test_dummy):
        raise AssertionError(
            f"Submission length ({len(submission_df)}) does not match input length ({len(df_test_dummy)})"
        )

    if list(submission_df.columns) != ["id", "after"]:
        raise AssertionError(f"Invalid submission columns: {submission_df.columns}")

    # Check for empty predictions
    if submission_df["after"].isnull().any():
        raise AssertionError("Submission contains null values!")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
