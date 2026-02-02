import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from unittest.mock import patch

# Import library components
from library.config import Config, set_seed
from library.utils import load_metadata
import library.utils
import library.hfbb_model
import library.tokenizer
import library.dataset
import library.inference
from library.hfbb_model import HFBBStats
from library.tokenizer import HybridTokenizer
from library.dataset import DatasetManager
from library.transformer_model import Seq2SeqTransformer, TransformerTrainer
from library.inference import HybridPredictor

# ==========================================
# 0. Setup and Monkey-Patching for Speed
# ==========================================
# To ensure this demo runs quickly, we intercept the data loading function
# and return small subsets of the real data. This allows us to test the
# full pipeline logic without waiting for millions of rows to process.

original_load_metadata = library.utils.load_metadata


def mocked_load_metadata(split="train"):
    """
    Mock function to load a small subset of data for demonstration purposes.
    """
    print(f"  [Demo] Loading subset of {split} data...")
    # Load the real file but only take the top N rows
    df = original_load_metadata(split)

    if split == "train":
        return df.head(5000).copy()  # Enough for stats and vocab
    elif split == "val":
        return df.head(500).copy()
    elif split == "test":
        return df.head(100).copy()  # Fast inference
    return df


# Apply the patch to all modules that import load_metadata
library.utils.load_metadata = mocked_load_metadata
library.hfbb_model.load_metadata = mocked_load_metadata
library.tokenizer.load_metadata = mocked_load_metadata
library.dataset.load_metadata = mocked_load_metadata
library.inference.load_metadata = mocked_load_metadata


def run_demo():
    # Set fixed seed for reproducibility
    set_seed(42)

    print("=== 1. Configuration Setup ===")
    # Initialize Config with demo-friendly parameters
    config = Config(debug=True, epochs=1, batch_size=16)
    config.idea_name = "demo_execution"  # Separate working directory

    # Clean up previous demo run if exists to ensure fresh start
    if os.path.exists(config.base_working_dir):
        shutil.rmtree(config.base_working_dir)

    config.print_summary()

    print("\n=== 2. Tier 1: HFBB (Statistical Model) ===")
    # Initialize and fit the Hierarchical Frequency Backoff model
    hfbb = HFBBStats(config)
    hfbb.fit(load_cached_data=False)  # Force re-compute on our subset

    # Verification: Query a known simple token (likely punctuation or common word)
    # We pick a token that likely exists in the first 5000 rows
    test_token = "."
    pred, conf = hfbb.query(test_token)
    print(f"  HFBB Query '{test_token}': Pred='{pred}', Conf={conf:.4f}")

    # Assert that we get a result (punctuation usually maps to itself with high confidence)
    if pred is not None:
        assert conf >= 0.0, "Confidence should be non-negative"
        print("  [✓] HFBB logic verified.")
    else:
        print(
            "  [!] HFBB did not find the token (might be absent in subset). Skipping assertion."
        )

    print("\n=== 3. Tokenizer Initialization ===")
    tokenizer = HybridTokenizer(config)
    tokenizer.fit(load_cached_data=False)

    # Verification: Check Vocab Sizes
    print(f"  Char Vocab Size: {tokenizer.char_vocab_size_actual}")
    print(f"  BPE Vocab Size: {tokenizer.bpe_vocab_size}")

    assert tokenizer.char_vocab_size_actual > 0, "Character vocab is empty!"
    # Encode/Decode check
    sample_text = "123"
    char_ids = tokenizer.encode_char(sample_text)
    bpe_ids = tokenizer.encode_bpe(sample_text)
    decoded_bpe = tokenizer.decode_bpe(bpe_ids)

    print(f"  Encoding '{sample_text}': CharIDs={char_ids}, BpeIDs={bpe_ids}")
    print(f"  Decoded BPE: '{decoded_bpe}'")

    assert len(char_ids) == len(sample_text), "Char encoding length mismatch"
    assert len(bpe_ids) > 0, "BPE encoding failed"
    print("  [✓] Tokenizer logic verified.")

    print("\n=== 4. Dataset & DataLoader ===")
    dataset_manager = DatasetManager(config, tokenizer)
    train_loader, val_loader = dataset_manager.get_dataloaders(load_cached_data=False)

    # Verification: Inspect one batch
    batch = next(iter(train_loader))
    enc_input, dec_target = batch

    print(f"  Batch Shapes - Encoder: {enc_input.shape}, Decoder: {dec_target.shape}")

    # Assertions on shapes
    # Encoder: [Batch, Seq_Len]
    assert (
        enc_input.shape[0] == config.batch_size
    ), f"Batch size mismatch: {enc_input.shape[0]}"
    assert dec_target.shape[0] == config.batch_size, "Target batch size mismatch"
    # Check for padding values
    assert (
        enc_input == tokenizer.char2id[tokenizer.PAD_TOKEN]
    ).any(), "Encoder batch should contain padding"
    print("  [✓] Data pipeline verified.")

    print("\n=== 5. Tier 2: Transformer Model & Training ===")
    # Initialize Model
    src_vocab = tokenizer.char_vocab_size_actual
    tgt_vocab = tokenizer.bpe_vocab_size
    src_pad = tokenizer.char2id[tokenizer.PAD_TOKEN]
    tgt_pad = tokenizer.bpe_pad_id

    model = Seq2SeqTransformer(config, src_vocab, tgt_vocab, src_pad, tgt_pad)

    # Verification: Forward Pass
    # Create dummy inputs on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_src = enc_input.to(device)
    dummy_tgt = dec_target.to(device)

    # Target input for forward is usually shifted or masked, but here we just check shape
    # Forward expects [Batch, Seq]
    with torch.no_grad():
        logits = model(dummy_src, dummy_tgt)

    print(f"  Output Logits Shape: {logits.shape}")
    assert logits.shape == (
        config.batch_size,
        dummy_tgt.shape[1],
        tgt_vocab,
    ), "Logits shape mismatch"

    # Train Loop
    print("  Starting Short Training Loop...")
    trainer = TransformerTrainer(config, model, train_loader, val_loader)
    trainer.train()

    # Verification: Checkpoint
    assert os.path.exists(trainer.best_model_path), "Best model checkpoint not found!"
    print("  [✓] Training loop and checkpointing verified.")

    print("\n=== 6. Inference & Submission ===")
    # Initialize Predictor
    # This re-loads the tokenizer and hfbb (from cache this time) and the best model
    predictor = HybridPredictor(config)

    # Run Generation
    predictor.generate_submission()

    # Verification: Submission File
    submission_path = config.submission_path
    assert os.path.exists(submission_path), "Submission file was not created!"

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission Head:\n{df_sub.head()}")

    # Check format
    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission columns missing"
    assert (
        len(df_sub) == 100
    ), f"Expected 100 predictions (from mocked test set), got {len(df_sub)}"

    # Check that predictions are strings
    assert (
        df_sub["after"].apply(lambda x: isinstance(x, str)).all()
    ), "All predictions must be strings"

    print("  [✓] Inference pipeline verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
