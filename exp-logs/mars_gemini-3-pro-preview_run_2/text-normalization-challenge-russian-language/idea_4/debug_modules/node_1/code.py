import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed, is_semiotic
from library.hfbb_engine import HFBBModel
from library.transformer_data import (
    prepare_dataloaders,
    CharTokenizer,
    NormalizationDataset,
)
from library.transformer_model import Seq2SeqTransformer
from library.trainer import fit_transformer
from library.inference import HybridSystem


def main():
    print("=== Starting Demonstration of Text Normalization Library ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.WORK_DIR = "./working/demo_execution/"
    Config.HFBB_CACHE_DIR = os.path.join(Config.WORK_DIR, "hfbb_cache")
    Config.TRANSFORMER_CACHE_DIR = os.path.join(Config.WORK_DIR, "transformer_cache")
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORK_DIR, "seq2seq_demo.pth")
    Config.VOCAB_PATH = os.path.join(Config.WORK_DIR, "vocab.json")
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")

    # Reduce compute load
    Config.DEBUG = True
    Config.DEBUG_SIZE = 2000  # Use only 2000 samples
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORK_DIR}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. HFBB Engine (Tier 1) Demonstration
    # ==========================================
    print("\n[2] Testing HFBB Model (Tier 1)...")

    hfbb = HFBBModel()

    # Fit on a small slice of data directly from the raw training file
    # We force re-computation to demonstrate the logic
    hfbb.fit(load_cached_data=False, max_rows=5000)

    # Validation: Check if maps are populated
    print(f"  Unigram Map Size: {len(hfbb.unigram_map)}")
    print(f"  Bigram Prev Map Size: {len(hfbb.bigram_prev_map)}")

    assert len(hfbb.unigram_map) > 0, "HFBB Unigram map should not be empty."

    # Validation: Check a specific lookup (Mocking a known pattern if possible, or just structure)
    # We pick a token that likely exists in the first 5000 rows, e.g., a punctuation or common word
    # Note: We can't guarantee specific content without reading the file, but we can check the API.
    test_token = list(hfbb.unigram_map.keys())[0]
    normalization = hfbb.get_normalization(test_token)
    print(f"  Lookup test: '{test_token}' -> '{normalization}'")
    assert normalization is not None, "HFBB lookup failed for a known key."

    # ==========================================
    # 3. Transformer Data Pipeline (Tier 2) Demonstration
    # ==========================================
    print("\n[3] Testing Transformer Data Pipeline...")

    # Prepare dataloaders (this triggers tokenization and dataset creation)
    # We disable caching to force processing
    train_loader, val_loader, tokenizer = prepare_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"  Vocab Size: {len(tokenizer)}")
    print(f"  Train Batches: {len(train_loader)}")

    # Validate Tokenizer
    sample_text = "hello 123"
    encoded = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"  Tokenizer Roundtrip: '{sample_text}' -> {encoded} -> '{decoded}'")

    # The tokenizer is character-level, so "hello 123" should be preserved exactly
    # unless some chars are UNK. Basic ASCII should be fine.
    assert decoded == sample_text, "Tokenizer decode did not match input."

    # Validate DataLoader Output
    src_batch, tgt_batch = next(iter(train_loader))
    print(f"  Source Batch Shape: {src_batch.shape}")
    print(f"  Target Batch Shape: {tgt_batch.shape}")

    assert src_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."
    assert src_batch.shape[1] == Config.MAX_SEQ_LEN, "Sequence length mismatch."

    # ==========================================
    # 4. Transformer Model Architecture Demonstration
    # ==========================================
    print("\n[4] Testing Transformer Model Architecture...")

    model = Seq2SeqTransformer(
        vocab_size=len(tokenizer),
        d_model=Config.D_MODEL,
        nhead=4,  # Reduced for demo
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
    ).to(Config.DEVICE)

    # Move batch to device
    src_batch = src_batch.to(Config.DEVICE)
    tgt_batch = tgt_batch.to(Config.DEVICE)

    # Forward pass (Teacher Forcing input: tgt without last token)
    tgt_input = tgt_batch[:, :-1]
    output = model(src_batch, tgt_input)

    print(f"  Model Output Shape: {output.shape}")

    # Output should be [Batch, Seq_Len - 1, Vocab]
    expected_seq_len = Config.MAX_SEQ_LEN - 1
    assert output.shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
        len(tokenizer),
    ), "Model output shape incorrect."

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[5] Testing Training Loop...")

    # We use the fit_transformer function which encapsulates the loop
    # Config is already set to 1 epoch and debug mode
    fit_transformer(load_cached_data=True, debug=True)

    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not created."
    print("  Training completed and checkpoint saved.")

    # ==========================================
    # 6. Inference System Demonstration
    # ==========================================
    print("\n[6] Testing Hybrid Inference System...")

    # Create a dummy test file to avoid processing the massive real test set
    mini_test_path = os.path.join(Config.WORK_DIR, "mini_test.csv")
    df_mini_test = pd.DataFrame(
        {
            "id": ["0_0", "0_1", "0_2", "1_0"],
            "sentence_id": [0, 0, 0, 1],
            "token_id": [0, 1, 2, 0],
            "before": ["The", "year", "1999", "Test"],
        }
    )
    df_mini_test.to_csv(mini_test_path, index=False)

    # Update Config to point to mini test
    Config.TEST_DATA = mini_test_path

    # Initialize System
    system = HybridSystem()

    # Run Generation
    # This will load the HFBB (re-fit/loaded) and the Transformer (checkpoint loaded)
    system.generate_submission()

    # Validate Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("  Submission Head:")
    print(df_sub.head())

    assert len(df_sub) == len(df_mini_test), "Submission row count mismatch."
    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission columns incorrect."

    # Check specific logic:
    # "The" -> Plain text, likely handled by HFBB or Identity fallback
    # "1999" -> Semiotic, likely handled by Transformer (or HFBB if frequent)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
