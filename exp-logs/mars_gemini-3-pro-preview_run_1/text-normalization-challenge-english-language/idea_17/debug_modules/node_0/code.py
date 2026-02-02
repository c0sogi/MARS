import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil

# 1. CONFIGURATION OVERRIDE
# We must modify the config before importing other library modules that rely on it.
import library.config as config

# Define demo paths
DEMO_WORKING_DIR = "./working/demo_execution"
DEMO_DATA_DIR = "./working/demo_data"
os.makedirs(DEMO_WORKING_DIR, exist_ok=True)
os.makedirs(DEMO_DATA_DIR, exist_ok=True)

# Override Config Paths
config.WORKING_DIR = DEMO_WORKING_DIR
config.CACHE_DIR = os.path.join(DEMO_WORKING_DIR, "cache")
config.CHECKPOINT_DIR = os.path.join(DEMO_WORKING_DIR, "checkpoints")
config.SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")
config.LOG_DIR = os.path.join(DEMO_WORKING_DIR, "logs")

config.TRAIN_DATA_PATH = os.path.join(DEMO_DATA_DIR, "train_subset.csv")
config.VAL_DATA_PATH = os.path.join(DEMO_DATA_DIR, "val_subset.csv")
config.TEST_DATA_PATH = os.path.join(DEMO_DATA_DIR, "test_subset.csv")
config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

# Override Training Hyperparameters for Speed
config.TAGGER_EPOCHS = 1
config.SEQ2SEQ_EPOCHS = 1
config.TAGGER_BATCH_SIZE = 32
config.SEQ2SEQ_BATCH_SIZE = 32
config.PLAIN_SUBSAMPLE_RATIO = 1.0  # We will manually balance the subset

# Ensure directories exist
for d in [
    config.CACHE_DIR,
    config.CHECKPOINT_DIR,
    config.SUBMISSION_DIR,
    config.LOG_DIR,
]:
    os.makedirs(d, exist_ok=True)

# Now import the rest of the library
from library.utils import set_seed
from library.vocabulary import build_vocabularies
from library.data_loader import get_dataloaders
from library.models import RegexBiLSTMTagger, CharLSTMSeq2Seq
from library.trainer import train_tagger, train_seq2seq
from library.inference import InferencePipeline


def create_demo_data():
    """
    Creates a small subset of the training data for demonstration purposes.
    Ensures a mix of PLAIN and non-PLAIN classes.
    """
    print("Creating demo data subsets...")

    # Load a chunk of the original training data
    source_train_path = "./metadata/train.csv"
    if not os.path.exists(source_train_path):
        raise FileNotFoundError(f"Source data not found at {source_train_path}")

    # Read first 50k rows to get enough variety
    df = pd.read_csv(source_train_path, nrows=50000, dtype=str, keep_default_na=False)

    # Filter for interesting classes (non-PLAIN/PUNCT)
    interesting = df[~df["class"].isin(["PLAIN", "PUNCT"])]
    boring = df[df["class"].isin(["PLAIN", "PUNCT"])]

    # Sample 500 interesting and 500 boring for training
    n_samples = 500
    train_subset = (
        pd.concat([interesting.head(n_samples), boring.head(n_samples)])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Create Val subset (next 100 rows)
    val_subset = (
        pd.concat(
            [
                interesting.iloc[n_samples : n_samples + 100],
                boring.iloc[n_samples : n_samples + 100],
            ]
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Create Test subset (structure: sentence_id, token_id, before, id)
    # We use the validation set as test input for simplicity
    test_subset = val_subset[["sentence_id", "token_id", "before", "id"]].copy()

    # Save to demo paths
    train_subset.to_csv(config.TRAIN_DATA_PATH, index=False)
    val_subset.to_csv(config.VAL_DATA_PATH, index=False)
    test_subset.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"Created train subset: {len(train_subset)} rows")
    print(f"Created val subset: {len(val_subset)} rows")
    print(f"Created test subset: {len(test_subset)} rows")


def run_demo():
    # Set seed
    set_seed(42)

    # 1. Prepare Data
    create_demo_data()

    # 2. Build Vocabularies
    # We set load_cached_data=False to force building from our new subset
    print("\n=== Building Vocabularies ===")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH, dtype=str, keep_default_na=False)
    vocab_words, vocab_chars, vocab_classes = build_vocabularies(
        df_train, load_cached_data=False
    )

    # Validation
    assert len(vocab_words) > 0, "Word vocabulary is empty"
    assert len(vocab_classes) > 0, "Class vocabulary is empty"

    # 3. Get DataLoaders
    print("\n=== Loading Data ===")
    loaders = get_dataloaders(
        vocab_words, vocab_chars, vocab_classes, load_cached_data=False
    )

    # 4. Initialize Models
    print("\n=== Initializing Models ===")
    tagger = RegexBiLSTMTagger(
        vocab_size_words=len(vocab_words),
        vocab_size_chars=len(vocab_chars),
        vocab_size_classes=len(vocab_classes),
    )

    sos_idx = vocab_chars.token2id[config.SOS_TOKEN]
    eos_idx = vocab_chars.token2id[config.EOS_TOKEN]

    seq2seq = CharLSTMSeq2Seq(
        vocab_size_chars=len(vocab_chars),
        vocab_size_classes=len(vocab_classes),
        sos_idx=sos_idx,
        eos_idx=eos_idx,
    )

    # 5. Train Tagger
    print("\n=== Training Tagger ===")
    train_tagger(
        model=tagger,
        train_loader=loaders["tagger_train"],
        val_loader=loaders["tagger_val"],
        vocab_classes_len=len(vocab_classes),
    )

    # Checkpoint validation
    tagger_ckpt = os.path.join(config.CHECKPOINT_DIR, "tagger_best_model.pth")
    assert os.path.exists(tagger_ckpt), "Tagger checkpoint was not saved"

    # 6. Train Seq2Seq
    print("\n=== Training Seq2Seq ===")
    train_seq2seq(
        model=seq2seq,
        train_loader=loaders["seq2seq_train"],
        val_loader=loaders["seq2seq_val"],
        vocab_chars_len=len(vocab_chars),
    )

    # Checkpoint validation
    seq2seq_ckpt = os.path.join(config.CHECKPOINT_DIR, "seq2seq_best_model.pth")
    assert os.path.exists(seq2seq_ckpt), "Seq2Seq checkpoint was not saved"

    # 7. Run Inference
    print("\n=== Running Inference Pipeline ===")
    # The pipeline reloads vocab and models from disk
    pipeline = InferencePipeline()
    pipeline.run()

    # 8. Validate Submission
    print("\n=== Validating Submission ===")
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission Rows: {len(df_sub)}")
    print(df_sub.head())

    # Check format
    expected_cols = ["id", "after"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check consistency with test input
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    assert len(df_sub) == len(
        df_test
    ), f"Submission length ({len(df_sub)}) does not match test set ({len(df_test)})"

    print("\nSUCCESS: Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
