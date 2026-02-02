import os
import sys
import pandas as pd
import torch
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Monkey Patch Config BEFORE importing other library modules
# This ensures all modules use the updated configuration
from library.config import Config

# Define demo paths
DEMO_DIR = "./working/demo_pipeline"
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(os.path.join(DEMO_DIR, "metadata"), exist_ok=True)
os.makedirs(os.path.join(DEMO_DIR, "cache"), exist_ok=True)
os.makedirs(os.path.join(DEMO_DIR, "vocabs"), exist_ok=True)
os.makedirs(os.path.join(DEMO_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(DEMO_DIR, "submission"), exist_ok=True)

# Override Config attributes
Config.WORKING_DIR = DEMO_DIR
Config.METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
Config.VOCAB_DIR = os.path.join(DEMO_DIR, "vocabs")
Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
Config.TRAIN_DATA_PATH = os.path.join(Config.METADATA_DIR, "train.csv")
Config.VAL_DATA_PATH = os.path.join(Config.METADATA_DIR, "val.csv")
Config.TEST_DATA_PATH = os.path.join(Config.METADATA_DIR, "test.csv")

# Optimization for Speed (Demo Settings)
Config.NUM_EPOCHS_TAGGER = 1
Config.NUM_EPOCHS_SEQ2SEQ = 1
Config.BATCH_SIZE = 16
Config.PLAIN_KEEP_RATE = 1.0  # Keep all data in our small subset
Config.PATIENCE_TAGGER = 1
Config.PATIENCE_SEQ2SEQ = 1
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead
Config.SEED = 42

# Import library modules
from library.dataset import build_artifacts, TaggerDataset, Seq2SeqDataset
from library.trainer import TaggerTrainer, Seq2SeqTrainer
from library.predictor import generate_submission
from library.features import process_dataset
from library.config import seed_everything


def prepare_demo_data():
    """
    Creates a small subset of the original metadata to allow for
    fast execution of the training and inference pipelines.
    """
    print("Preparing demo dataset...")

    # Source paths (from the provided metadata)
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Read small chunks
    # We take 1000 rows to ensure we have enough diversity for vocabs
    df_train = pd.read_csv(orig_train_path, nrows=1000)
    df_val = pd.read_csv(orig_val_path, nrows=200)
    df_test = pd.read_csv(orig_test_path, nrows=200)

    # Save to demo directory
    df_train.to_csv(Config.TRAIN_DATA_PATH, index=False)
    df_val.to_csv(Config.VAL_DATA_PATH, index=False)
    df_test.to_csv(Config.TEST_DATA_PATH, index=False)

    print(f"Demo data saved to {Config.METADATA_DIR}")


def run_demo():
    seed_everything(Config.SEED)

    # 1. Prepare Data
    prepare_demo_data()

    # 2. Build Artifacts (Vocabs, Knowledge Base)
    print("\n[Step 1] Building Artifacts...")
    build_artifacts(load_cached_data=False)

    # Verify artifacts
    vocab_words = os.path.join(Config.VOCAB_DIR, "vocab_words.json")
    vocab_classes = os.path.join(Config.VOCAB_DIR, "vocab_classes.json")
    kb_path = os.path.join(Config.CACHE_DIR, "knowledge_base.parquet")

    if not os.path.exists(vocab_words):
        raise FileNotFoundError("Failed to build vocab_words.json")
    if not os.path.exists(kb_path):
        raise FileNotFoundError("Failed to build knowledge_base.parquet")
    print("Artifacts built successfully.")

    # 3. Feature Extraction
    # We explicitly run this for 'train' to generate vocab_chars.json
    # before the TaggerTrainer tries to load it.
    print("\n[Step 2] Extracting Features...")
    process_dataset("train", load_cached_data=False)

    # Verify feature cache
    train_feats = os.path.join(Config.CACHE_DIR, "train_char_features.npy")
    vocab_chars = os.path.join(Config.VOCAB_DIR, "vocab_chars.json")

    if not os.path.exists(train_feats):
        raise FileNotFoundError("Failed to cache train features")
    if not os.path.exists(vocab_chars):
        raise FileNotFoundError("Failed to generate vocab_chars.json")
    print("Features extracted and cached.")

    # 4. Train Tagger
    print("\n[Step 3] Training Tagger Model...")
    tagger_trainer = TaggerTrainer()
    tagger_trainer.train()

    # Verify checkpoint
    tagger_ckpt = os.path.join(Config.CHECKPOINT_DIR, "tagger_best_model.pth")
    if not os.path.exists(tagger_ckpt):
        raise FileNotFoundError("Tagger model checkpoint not created.")
    print("Tagger training complete.")

    # 5. Train Seq2Seq
    print("\n[Step 4] Training Seq2Seq Model...")
    seq2seq_trainer = Seq2SeqTrainer()
    seq2seq_trainer.train()

    # Verify checkpoint
    seq2seq_ckpt = os.path.join(Config.CHECKPOINT_DIR, "seq2seq_best_model.pth")
    if not os.path.exists(seq2seq_ckpt):
        raise FileNotFoundError("Seq2Seq model checkpoint not created.")
    print("Seq2Seq training complete.")

    # 6. Inference
    print("\n[Step 5] Running Inference Pipeline...")
    generate_submission()

    # Verify submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check format
    expected_cols = ["id", "after"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
