import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch

# Import library modules
import library.config
from library.utils import set_seed, get_device
from library.features import ExplicitFeatureExtractor
from library.data_processing import prepare_data, TaggerDataset, Seq2SeqDataset
from library.models_tagger import QuadHybridBiLSTM
from library.models_seq2seq import CharTransformer
from library.train_tagger import train_tagger_model
from library.train_seq2seq import train_seq2seq_model
from library.inference import generate_submission

# =========================================================================
# 1. Configuration Monkey-Patching for Speed
# =========================================================================
# We patch the Config class to force low-resource usage for this demo.
original_init = library.config.Config.__init__


def patched_init(self, debug=True):
    # Initialize with debug=True to trigger some internal reductions
    original_init(self, debug=True)

    # Force specific parameters for ultra-fast execution
    self.NUM_EPOCHS = 1
    self.SAMPLE_SIZE = 200  # Only use 200 rows
    self.BATCH_SIZE = 4
    self.PATIENCE = 1

    # Redirect working directory to avoid conflicts
    self.WORKING_DIR = "./working/demo_execution"
    self.SUBMISSION_DIR = os.path.join(self.WORKING_DIR, "submission")

    # Update dependent paths
    os.makedirs(self.WORKING_DIR, exist_ok=True)
    os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

    self.TAGGER_MODEL_PATH = os.path.join(self.WORKING_DIR, "tagger_demo.pth")
    self.SEQ2SEQ_MODEL_PATH = os.path.join(self.WORKING_DIR, "seq2seq_demo.pth")
    self.KB_PATH = os.path.join(self.WORKING_DIR, "knowledge_base.parquet")
    self.BPE_MODEL_PREFIX = os.path.join(self.WORKING_DIR, "bpe_tokenizer")
    self.VOCAB_DIR = os.path.join(self.WORKING_DIR, "vocabs")
    self.CACHE_DIR = os.path.join(self.WORKING_DIR, "cache")
    self.SUBMISSION_FILE = os.path.join(self.SUBMISSION_DIR, "submission.csv")

    # Ensure dirs exist
    os.makedirs(self.VOCAB_DIR, exist_ok=True)
    os.makedirs(self.CACHE_DIR, exist_ok=True)


# Apply the patch
library.config.Config.__init__ = patched_init

# =========================================================================
# 2. Demo Functions
# =========================================================================


def demo_feature_extraction():
    print("\n=== Demo: Explicit Feature Extraction ===")
    extractor = ExplicitFeatureExtractor()

    # Test Case: Currency and Decimal
    token = "$3.16"
    features = extractor.transform_single(token)

    print(f"Token: {token}")
    print(f"Feature Vector Shape: {features.shape}")
    print(f"Feature Vector: {features}")

    # Validation
    # Index 1: Is Decimal (^\d*\.\d+$) -> 3.16 matches part of it, but regex is strict ^$.
    # Wait, $3.16 does NOT match ^\d*\.\d+$. It matches currency symbol.
    # Let's check specific indices based on library/features.py
    # Index 8: Is Punctuation (^[^\w\s]+$) -> No ($ is punct, 3 is not)
    # Index 9: Has Currency Symbol ([$£€¥¢]) -> Yes ($)
    # Index 13: Has Digit (\d) -> Yes

    # Index 9 corresponds to 'Has Currency Symbol'
    assert (
        features[9] == 1.0
    ), "Feature extraction failed: Currency symbol not detected."
    # Index 13 corresponds to 'Has Digit'
    assert features[13] == 1.0, "Feature extraction failed: Digit not detected."

    print("Feature extraction logic verified.")


def demo_data_processing():
    print("\n=== Demo: Data Processing ===")
    # This calls prepare_data which handles vocabs, BPE, and grouping
    # Since we patched Config, it uses SAMPLE_SIZE=200
    artifacts = prepare_data(load_cached_data=False)

    print("Keys in artifacts:", artifacts.keys())

    # Validate Vocabs
    vocab_words = artifacts["vocab_words"]
    print(f"Word Vocab Size: {len(vocab_words)}")
    assert len(vocab_words) > 0, "Word vocabulary is empty."

    # Validate Grouped Data
    train_grouped = artifacts["train_grouped"]
    print(f"Train Grouped Rows (Sentences): {len(train_grouped)}")
    # We expect some sentences. Since we sampled 200 rows (tokens), sentences will be fewer.
    assert len(train_grouped) > 0, "Train grouped dataframe is empty."
    assert "before" in train_grouped.columns, "Missing 'before' column in grouped data."

    return artifacts


def demo_model_architectures(artifacts):
    print("\n=== Demo: Model Architectures ===")
    config = library.config.Config()
    device = get_device()

    vocab_words = artifacts["vocab_words"]
    vocab_chars = artifacts["vocab_chars"]
    vocab_classes = artifacts["vocab_classes"]

    # --- Tagger Model ---
    print("Initializing QuadHybridBiLSTM Tagger...")
    tagger = QuadHybridBiLSTM(
        num_classes=len(vocab_classes),
        vocab_words=vocab_words,
        vocab_chars=vocab_chars,
        vocab_bpe_size=config.BPE_VOCAB_SIZE,
    ).to(device)

    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    char_len = config.MAX_TOKEN_CHAR_LEN
    bpe_len = 10  # heuristic from dataset class

    word_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)
    char_ids = torch.zeros((batch_size, seq_len, char_len), dtype=torch.long).to(device)
    bpe_ids = torch.zeros((batch_size, seq_len, bpe_len), dtype=torch.long).to(device)
    features = torch.zeros(
        (batch_size, seq_len, config.NUM_REGEX_FEATURES), dtype=torch.float32
    ).to(device)

    # Forward Pass
    logits = tagger(word_ids, char_ids, bpe_ids, features)
    print(f"Tagger Output Shape: {logits.shape}")

    assert logits.shape == (
        batch_size,
        seq_len,
        len(vocab_classes),
    ), f"Tagger output shape mismatch. Expected {(batch_size, seq_len, len(vocab_classes))}, got {logits.shape}"

    # --- Seq2Seq Model ---
    print("Initializing CharTransformer Seq2Seq...")
    seq2seq = CharTransformer(
        vocab_chars_size=len(vocab_chars), vocab_classes_size=len(vocab_classes)
    ).to(device)

    # Dummy inputs
    tgt_len = 15
    src_ids = torch.zeros((batch_size, char_len), dtype=torch.long).to(device)
    tgt_in = torch.zeros((batch_size, tgt_len), dtype=torch.long).to(device)
    class_id = torch.zeros((batch_size,), dtype=torch.long).to(device)

    # Forward Pass
    seq_logits = seq2seq(src_ids, tgt_in, class_id)
    print(f"Seq2Seq Output Shape: {seq_logits.shape}")

    assert seq_logits.shape == (
        batch_size,
        tgt_len,
        len(vocab_chars),
    ), f"Seq2Seq output shape mismatch. Expected {(batch_size, tgt_len, len(vocab_chars))}, got {seq_logits.shape}"

    print("Model architectures verified.")


def demo_training():
    print("\n=== Demo: Training Loops ===")
    # We rely on the patched config to keep this short (1 epoch, small data)

    print(">> Training Tagger...")
    # Force recompute to ensure we use the sampled data
    train_tagger_model(load_cached_data=False)

    config = library.config.Config()
    assert os.path.exists(config.TAGGER_MODEL_PATH), "Tagger model file was not saved."

    print(">> Training Seq2Seq...")
    train_seq2seq_model(load_cached_data=True)  # Can reuse cache from tagger step

    assert os.path.exists(
        config.SEQ2SEQ_MODEL_PATH
    ), "Seq2Seq model file was not saved."
    print("Training loops completed successfully.")


def demo_inference():
    print("\n=== Demo: Inference & Submission ===")

    # Run the full inference pipeline
    # This loads the models saved in demo_training and generates submission.csv
    generate_submission()

    config = library.config.Config()
    submission_path = config.SUBMISSION_FILE

    if os.path.exists(submission_path):
        df = pd.read_csv(submission_path)
        print(f"Submission file created at {submission_path}")
        print(f"Rows: {len(df)}")
        print("Head:")
        print(df.head())

        assert len(df) > 0, "Submission file is empty."
        assert (
            "id" in df.columns and "after" in df.columns
        ), "Submission file missing required columns."
    else:
        raise FileNotFoundError("Submission file was not created.")


# =========================================================================
# Main Execution
# =========================================================================

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # 1. Feature Extraction Logic
    demo_feature_extraction()

    # 2. Data Processing
    artifacts = demo_data_processing()

    # 3. Model Architecture Validation
    demo_model_architectures(artifacts)

    # 4. Training Simulation
    demo_training()

    # 5. Inference Simulation
    demo_inference()

    print("\nAll demos completed successfully.")
