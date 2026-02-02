import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import time

# Import library modules
from library.config import Config
from library.utils import set_seed, get_device
from library.tokenization import build_tokenizers
from library.data_manager import get_dataloaders, SemioticDataset
from library.hfbb_layer import HFBBModel
from library.transformer_arch import CharToSubwordTransformer
from library.trainer import train_model
from library.predictor import InferenceEngine


def create_mini_datasets(n_samples=500):
    """
    Creates small subsets of the original data for demonstration purposes.
    """
    print(f"Creating mini datasets with {n_samples} samples...")

    # Read original metadata
    df_train = pd.read_csv("./metadata/train.csv", nrows=n_samples)
    df_val = pd.read_csv("./metadata/val.csv", nrows=n_samples)
    df_test = pd.read_csv("./metadata/test.csv", nrows=n_samples)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define paths for mini datasets
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Save mini datasets
    df_train.to_csv(mini_train_path, index=False)
    df_val.to_csv(mini_val_path, index=False)
    df_test.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def configure_demo_environment(mini_train_path, mini_val_path, mini_test_path):
    """
    Overrides Config parameters for a fast demo run.
    """
    print("Overriding Config parameters for speed...")

    # Paths
    Config.TRAIN_DATA_PATH = mini_train_path
    Config.VAL_DATA_PATH = mini_val_path
    Config.TEST_DATA_PATH = mini_test_path

    # Redirect caches to demo folder to avoid conflicts
    Config.HFBB_CACHE_DIR = os.path.join(Config.WORKING_DIR, "hfbb_cache")
    Config.TRANSFORMER_CACHE_DIR = os.path.join(Config.WORKING_DIR, "data_cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.TOKENIZER_DIR = os.path.join(Config.WORKING_DIR, "tokenizers")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update derived paths
    Config.HFBB_UNIGRAM_PATH = os.path.join(Config.HFBB_CACHE_DIR, "unigram.parquet")
    Config.HFBB_BIGRAM_PREV_PATH = os.path.join(
        Config.HFBB_CACHE_DIR, "bigram_prev.parquet"
    )
    Config.HFBB_BIGRAM_NEXT_PATH = os.path.join(
        Config.HFBB_CACHE_DIR, "bigram_next.parquet"
    )
    Config.HFBB_TRIGRAM_PATH = os.path.join(Config.HFBB_CACHE_DIR, "trigram.parquet")

    Config.TRANSFORMER_TRAIN_PATH = os.path.join(
        Config.TRANSFORMER_CACHE_DIR, "train_proc.parquet"
    )
    Config.TRANSFORMER_VAL_PATH = os.path.join(
        Config.TRANSFORMER_CACHE_DIR, "val_proc.parquet"
    )

    Config.CHAR_VOCAB_PATH = os.path.join(Config.TOKENIZER_DIR, "char_vocab.json")
    Config.BPE_MODEL_PREFIX = os.path.join(Config.TOKENIZER_DIR, "bpe_demo")

    Config.BEST_MODEL_PATH = os.path.join(
        Config.CHECKPOINT_DIR, "transformer_demo_best.pth"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create new directories
    Config.setup()

    # Model Hyperparameters (Tiny model for speed)
    Config.D_MODEL = 64
    Config.NHEAD = 2
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 128
    Config.DROPOUT = 0.0

    # Training Hyperparameters
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.WARMUP_STEPS = 5
    Config.DEBUG_SUBSET_SIZE = 0  # We are already using a mini dataset

    # Tokenizer
    Config.BPE_VOCAB_SIZE = 500  # Small vocab for small data


def demo_tokenization():
    print("\n=== Demo: Tokenization ===")
    # Build tokenizers from scratch (load_cached_data=False)
    char_tokenizer, bpe_tokenizer = build_tokenizers(load_cached_data=False)

    print(f"Char Tokenizer Vocab Size: {len(char_tokenizer)}")
    print(f"BPE Tokenizer Vocab Size: {len(bpe_tokenizer)}")

    # Verification
    assert len(char_tokenizer) > 0, "Char tokenizer vocab is empty"
    assert len(bpe_tokenizer) > 0, "BPE tokenizer vocab is empty"

    # Test Encoding/Decoding
    test_str = "abc 123"
    ids = char_tokenizer.encode(test_str)
    decoded = char_tokenizer.decode(ids)
    # Note: CharTokenizer might treat space as UNK if not in training data, or if space handling is specific.
    # In this dataset, raw text usually has spaces.
    print(f"Original: '{test_str}' -> Decoded: '{decoded}'")

    return char_tokenizer, bpe_tokenizer


def demo_hfbb():
    print("\n=== Demo: HFBB (Tier 1) ===")
    # Build HFBB model from scratch using the mini dataset
    hfbb = HFBBModel(load_cached_data=False)

    # Verify internal maps are populated
    print(f"Unigram Map Size: {len(hfbb.unigram_map)}")
    assert hasattr(hfbb, "unigram_map"), "HFBB model missing unigram map"

    # Test Query (using a token likely to be in the mini set, or just checking logic)
    # We grab a real example from the mini train set to ensure a hit
    df_mini = pd.read_csv(Config.TRAIN_DATA_PATH)
    if not df_mini.empty:
        row = df_mini.iloc[0]
        before = str(row["before"])
        after = str(row["after"])

        # If confidence is high enough, it should predict
        # We can't guarantee confidence on such small data, but we can check the method runs
        pred = hfbb.query(before, "", "")
        print(f"Query '{before}': {pred} (Expected: {after})")

        # Validation: The method should return a string or None, not crash
        assert pred is None or isinstance(pred, str)


def demo_transformer_training(char_tokenizer, bpe_tokenizer):
    print("\n=== Demo: Transformer Training (Tier 2) ===")

    # 1. Verify Data Loading
    train_loader, val_loader = get_dataloaders(
        char_tokenizer, bpe_tokenizer, load_cached_data=False
    )
    batch = next(iter(train_loader))
    src, tgt = batch
    print(f"Batch Shapes - Src: {src.shape}, Tgt: {tgt.shape}")

    assert (
        src.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {src.shape[0]}"
    assert src.shape[1] <= Config.MAX_SRC_LEN, "Source sequence too long"

    # 2. Verify Model Architecture
    model = CharToSubwordTransformer(
        src_vocab_size=len(char_tokenizer),
        tgt_vocab_size=len(bpe_tokenizer),
        src_pad_idx=char_tokenizer.pad_token_id,
        tgt_pad_idx=bpe_tokenizer.pad_token_id,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
    ).to(get_device())

    # Forward pass check
    src_dev = src.to(get_device())
    tgt_dev = tgt.to(get_device())
    tgt_input = tgt_dev[:, :-1]  # Shift for teacher forcing

    logits = model(src_dev, tgt_input)
    print(f"Logits Shape: {logits.shape}")  # Should be [Batch, SeqLen-1, Vocab]

    assert logits.shape == (Config.BATCH_SIZE, tgt_input.shape[1], len(bpe_tokenizer))

    # 3. Run Training Loop
    print("Starting training loop...")
    trained_model = train_model(
        char_tokenizer, bpe_tokenizer, num_epochs=Config.NUM_EPOCHS
    )

    # Verify Checkpoint
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint not found after training"
    print("Training completed successfully.")


def demo_inference():
    print("\n=== Demo: Inference Pipeline ===")

    # Initialize Engine (loads tokenizers and models from cache/checkpoints created above)
    engine = InferenceEngine(load_cached_data=True)

    # Generate Submission
    submission_df = engine.generate_submission()

    print(f"Submission Shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())

    # Validation
    assert "id" in submission_df.columns
    assert "after" in submission_df.columns
    assert len(submission_df) > 0
    assert os.path.exists(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    Config.WORKING_DIR = "./working/demo_execution"

    # 2. Prepare Data & Config
    mini_train, mini_val, mini_test = create_mini_datasets(n_samples=500)
    configure_demo_environment(mini_train, mini_val, mini_test)

    # 3. Run Demos
    # Tokenization
    char_tok, bpe_tok = demo_tokenization()

    # HFBB
    demo_hfbb()

    # Transformer Training
    demo_transformer_training(char_tok, bpe_tok)

    # Inference
    demo_inference()

    print("\nAll demonstrations completed successfully.")
