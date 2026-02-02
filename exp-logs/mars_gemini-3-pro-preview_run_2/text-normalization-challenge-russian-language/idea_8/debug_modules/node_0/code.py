import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.hfbb_engine import HFBBModel
from library.text_processing import build_tokenizers, CharTokenizer, TargetTokenizer
from library.dataset_builder import DatasetBuilder
from library.trainer import train_model
from library.inference import HybridPredictor, generate_submission


def main():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print(">>> [1/7] Setting up environment and overriding configuration...")

    # Define a specific directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Define paths for mini-datasets
    MINI_TRAIN_PATH = os.path.join(DEMO_DIR, "mini_train.csv")
    MINI_VAL_PATH = os.path.join(DEMO_DIR, "mini_val.csv")
    MINI_TEST_PATH = os.path.join(DEMO_DIR, "mini_test.csv")

    # Patch the Config class to use the demo directory and settings
    # We must update all dependent paths since they were initialized at import time
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_FILE = MINI_TRAIN_PATH
    Config.VAL_FILE = MINI_VAL_PATH
    Config.TEST_FILE = MINI_TEST_PATH

    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.HFBB_CACHE_DIR = os.path.join(DEMO_DIR, "hfbb_cache")
    Config.HFBB_UNIGRAM_PATH = os.path.join(Config.HFBB_CACHE_DIR, "unigram.parquet")
    Config.HFBB_BIGRAM_PREV_PATH = os.path.join(
        Config.HFBB_CACHE_DIR, "bigram_prev.parquet"
    )
    Config.HFBB_BIGRAM_NEXT_PATH = os.path.join(
        Config.HFBB_CACHE_DIR, "bigram_next.parquet"
    )
    Config.HFBB_TRIGRAM_PATH = os.path.join(Config.HFBB_CACHE_DIR, "trigram.parquet")

    Config.DATA_CACHE_DIR = os.path.join(DEMO_DIR, "data_cache")
    Config.RESIDUAL_TRAIN_PATH = os.path.join(
        Config.DATA_CACHE_DIR, "enriched_residual_train.parquet"
    )
    Config.RESIDUAL_VAL_PATH = os.path.join(
        Config.DATA_CACHE_DIR, "enriched_residual_val.parquet"
    )

    Config.TOKENIZER_DIR = os.path.join(DEMO_DIR, "tokenizers")
    Config.BPE_MODEL_PREFIX = os.path.join(Config.TOKENIZER_DIR, "bpe_demo")
    Config.BPE_MODEL_PATH = f"{Config.BPE_MODEL_PREFIX}.model"
    Config.BPE_VOCAB_PATH = f"{Config.BPE_MODEL_PREFIX}.vocab"
    Config.CHAR_VOCAB_PATH = os.path.join(Config.TOKENIZER_DIR, "char_vocab.json")

    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.MODEL_BEST_PATH = os.path.join(
        Config.CHECKPOINT_DIR, "transformer_demo_best.pth"
    )

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.N_FOLDS = 2  # Minimum for jackknife
    Config.BPE_VOCAB_SIZE = 1000  # Small vocab for small data
    Config.WARMUP_STEPS = 5
    Config.DEBUG = True

    # Create directories
    Config.setup_dirs()
    set_seed(42)

    # ==========================================
    # 2. Data Preparation (Mini-Subsets)
    # ==========================================
    print(">>> [2/7] Creating mini-datasets...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take small subsets (ensure we don't split sentences in a weird way, though GroupShuffleSplit handled that upstream)
    # We take enough rows to ensure some variety
    mini_train = orig_train.head(3000).copy()
    mini_val = orig_val.head(600).copy()
    mini_test = orig_test.head(700).copy()

    # Save to demo location
    mini_train.to_csv(MINI_TRAIN_PATH, index=False)
    mini_val.to_csv(MINI_VAL_PATH, index=False)
    mini_test.to_csv(MINI_TEST_PATH, index=False)

    print(f"    Train size: {len(mini_train)}")
    print(f"    Val size: {len(mini_val)}")
    print(f"    Test size: {len(mini_test)}")

    # ==========================================
    # 3. HFBB Model (Statistical Layer)
    # ==========================================
    print(">>> [3/7] Building and testing HFBB Model...")

    hfbb = HFBBModel()
    # Fit from scratch on mini_train
    hfbb.fit()

    # Verify files were created
    assert os.path.exists(Config.HFBB_UNIGRAM_PATH), "HFBB Unigram cache missing"

    # Load and Query
    hfbb.load()

    # Test query on a known token from the mini set
    # We pick a token that likely exists in the first 3000 rows, e.g., a punctuation or common word
    sample_token = mini_train.iloc[0]["before"]
    sample_res = hfbb.query(sample_token)

    print(f"    Query '{sample_token}': {sample_res}")
    assert sample_res["pred"] is not None, "HFBB should predict known token"
    assert sample_res["source"] in [
        "UNIGRAM",
        "BIGRAM_NEXT",
        "BIGRAM_PREV",
        "TRIGRAM",
    ], "Invalid HFBB source"

    # ==========================================
    # 4. Tokenizers
    # ==========================================
    print(">>> [4/7] Building Tokenizers...")

    # Build tokenizers using the mini training data
    char_tokenizer, target_tokenizer = build_tokenizers(
        train_df=mini_train, load_cached_data=False
    )

    # Verification
    assert os.path.exists(Config.CHAR_VOCAB_PATH), "Char vocab file missing"
    assert os.path.exists(Config.BPE_MODEL_PATH), "BPE model file missing"

    # Test Char Tokenizer
    test_str = "abc 123"
    encoded_chars = char_tokenizer.encode(test_str, add_special_tokens=True)
    decoded_chars = char_tokenizer.decode(encoded_chars, remove_special_tokens=True)
    print(f"    Char Tokenizer: '{test_str}' -> {encoded_chars} -> '{decoded_chars}'")
    # Note: Decoding might lose info if chars are UNK, but for basic ascii it should work if vocab covers it

    # Test Target Tokenizer
    test_tgt = "hello world"
    encoded_tgt = target_tokenizer.encode(test_tgt)
    decoded_tgt = target_tokenizer.decode(encoded_tgt)
    print(f"    Target Tokenizer: '{test_tgt}' -> {encoded_tgt} -> '{decoded_tgt}'")

    # ==========================================
    # 5. Dataset Builder (Curriculum/Residuals)
    # ==========================================
    print(">>> [5/7] Building Residual Dataset (Jackknifing)...")

    builder = DatasetBuilder()
    # Force build from scratch
    train_res_df, val_res_df = builder.build_dataset(load_cached_data=False)

    print(f"    Residual Train Size: {len(train_res_df)}")
    print(f"    Residual Val Size: {len(val_res_df)}")

    assert "input_text" in train_res_df.columns
    assert "target_text" in train_res_df.columns
    assert os.path.exists(Config.RESIDUAL_TRAIN_PATH)

    # ==========================================
    # 6. Transformer Training
    # ==========================================
    print(">>> [6/7] Training Transformer Model...")

    # Train model (will load the cached datasets we just built)
    model = train_model(load_cached_data=True)

    assert isinstance(model, nn.Module)
    assert os.path.exists(Config.MODEL_BEST_PATH), "Best model checkpoint not found"

    # ==========================================
    # 7. Inference & Submission
    # ==========================================
    print(">>> [7/7] Running Inference and Generating Submission...")

    # Initialize Predictor
    predictor = HybridPredictor()
    predictor.load_resources()

    # Run prediction on mini test set
    preds = predictor.predict(mini_test)

    assert len(preds) == len(mini_test), "Prediction count mismatch"

    # Generate submission file
    submission = generate_submission(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"
    assert len(submission) == len(mini_test)
    assert list(submission.columns) == ["id", "after"]

    print("\n>>> Demo execution completed successfully!")
    print(f"Output directory: {DEMO_DIR}")


if __name__ == "__main__":
    main()
