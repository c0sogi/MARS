import os
import pandas as pd
import torch
import shutil
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_data, is_semiotic
from library.hfbb import HFBBEngine
from library.tokenizers import HeterogeneousTokenizer
from library.data_factory import get_dataloaders
from library.model import CharToSubwordTransformer
from library.train_eval import Trainer
from library.pipeline import HybridInference


def create_mini_datasets(n_samples=500):
    """
    Creates small subsets of the original data for demonstration purposes.
    Ensures we have both PLAIN and SEMIOTIC classes to test all pipeline components.
    """
    print(f"\n[Demo] Creating mini datasets (N={n_samples})...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Load a chunk of the real training data
    # We need to ensure we get some non-PLAIN classes for the neural net
    df_full = pd.read_csv(Config.TRAIN_FILE, nrows=n_samples * 10)

    # Separate plain and semiotic to ensure representation
    df_semiotic = df_full[df_full["class"].isin(Config.SEMIOTIC_CLASSES)]
    df_plain = df_full[~df_full["class"].isin(Config.SEMIOTIC_CLASSES)]

    # Take a mix
    n_sem = min(len(df_semiotic), n_samples // 2)
    n_plain = n_samples - n_sem

    df_mini_train = pd.concat([df_semiotic.head(n_sem), df_plain.head(n_plain)])
    # Shuffle
    df_mini_train = df_mini_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create mini val (using same strategy or just tail)
    df_mini_val = df_full.tail(n_samples).reset_index(drop=True)

    # Create mini test (drop 'after' and 'class')
    df_mini_test = df_full.head(n_samples)[["sentence_id", "token_id", "before"]].copy()

    # Save
    df_mini_train.to_csv(mini_train_path, index=False)
    df_mini_val.to_csv(mini_val_path, index=False)
    df_mini_test.to_csv(mini_test_path, index=False)

    print(f"  Mini Train: {len(df_mini_train)} rows")
    print(f"  Mini Val: {len(df_mini_val)} rows")
    print(f"  Mini Test: {len(df_mini_test)} rows")

    return mini_train_path, mini_val_path, mini_test_path


def configure_demo_environment():
    """
    Overrides Config settings to use a temporary directory and speed up execution.
    """
    print("[Demo] Configuring environment...")

    # Setup demo working directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.HFBB_CACHE_DIR = os.path.join(demo_dir, "hfbb_cache")
    Config.TRANSFORMER_CACHE_DIR = os.path.join(demo_dir, "transformer_cache")

    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "seq2seq_demo.pth")

    Config.BPE_MODEL_PREFIX = os.path.join(demo_dir, "bpe_demo")
    Config.VOCAB_PATH = os.path.join(demo_dir, "vocab.json")

    # Setup directories based on new paths
    Config.setup_directories()

    # Override Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.UPSAMPLE_TARGET_COUNT = 20  # Minimal upsampling
    Config.BPE_VOCAB_SIZE = 300  # Small vocab for small dataset

    # Smaller Model for Speed
    Config.ENC_EMB_DIM = 32
    Config.ENC_HIDDEN_DIM = 64
    Config.ENC_LAYERS = 2
    Config.ENC_HEADS = 4
    Config.DEC_EMB_DIM = 32
    Config.DEC_HIDDEN_DIM = 64
    Config.DEC_LAYERS = 2
    Config.DEC_HEADS = 4
    Config.MAX_SEQ_LEN = 32

    # Set device
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {Config.DEVICE}")


def demo_hfbb_engine():
    """
    Demonstrates Tier 1: HFBB Engine (Memorization).
    """
    print("\n[Demo] Testing HFBB Engine...")

    # Initialize
    hfbb = HFBBEngine()

    # Load mini train data
    df_train = load_data("train")

    # Fit (this will compute maps and save cache)
    hfbb.fit(df_train, load_cached_data=False)

    # Verification
    # Pick a sample from the dataframe to query
    sample = df_train.iloc[0]
    token = sample["before"]
    # We need context. Since we just loaded the DF, we can manually grab prev/next if we sorted,
    # but let's just test the Unigram fallback which is easiest to verify without reconstructing context columns manually here.
    # The HFBB fit method computes context internally.

    # Let's verify files were created
    assert os.path.exists(hfbb.cache_files["unigram"]), "Unigram cache not found"
    assert len(hfbb.unigram_map) > 0, "Unigram map is empty"

    print("  HFBB fit complete and verified.")


def demo_tokenizer():
    """
    Demonstrates Tokenizer (Char source, BPE target).
    """
    print("\n[Demo] Testing Heterogeneous Tokenizer...")

    tokenizer = HeterogeneousTokenizer()
    df_train = load_data("train")

    # Fit
    tokenizer.fit(df_train, load_cached_data=False)

    # Test Encoding
    test_str = "123"
    src_ids = tokenizer.encode_source(test_str)
    print(f"  Source '{test_str}' -> IDs: {src_ids}")

    # Verify Source Encoding (Char level)
    # Should have length equal to string length (unless unk/special handling differs, but usually 1-to-1 for chars)
    assert len(src_ids) == len(test_str), "Source encoding length mismatch"

    # Test Target Encoding (BPE)
    tgt_str = "сто двадцать три"
    tgt_ids = tokenizer.encode_target(tgt_str)
    decoded = tokenizer.decode_target(tgt_ids)
    print(f"  Target '{tgt_str}' -> IDs: {tgt_ids} -> Decoded: '{decoded}'")

    # Verify Reconstruction
    # Note: BPE might normalize spaces or casing, but content should match
    assert decoded == tgt_str, "Target decoding failed to reconstruct input"

    return tokenizer


def demo_training_pipeline(tokenizer):
    """
    Demonstrates the Neural Network Training Pipeline.
    """
    print("\n[Demo] Testing Training Pipeline...")

    # Initialize Trainer
    trainer = Trainer(device=Config.DEVICE)

    # Inject the already fitted tokenizer to save time (optional, as Trainer loads it)
    trainer.tokenizer = tokenizer

    # Run Training
    # This uses the mini dataset and reduced epochs defined in configure_demo_environment
    model = trainer.run(epochs=Config.EPOCHS, load_cached_data=False)

    # Verify Model Output
    assert isinstance(
        model, CharToSubwordTransformer
    ), "Trainer did not return a Transformer model"
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model checkpoint not found"

    print("  Training complete. Model saved.")


def demo_inference_pipeline():
    """
    Demonstrates the Hybrid Inference Pipeline.
    """
    print("\n[Demo] Testing Inference Pipeline...")

    # Initialize Inference Engine
    inference = HybridInference(device=Config.DEVICE)

    # Load Resources (HFBB, Tokenizer, Model)
    # We set load_cached_data=True because we just created them in previous steps
    inference.load_resources(load_cached_data=True)

    # Load Test Data
    df_test = load_data("test")

    # Run Prediction
    submission = inference.predict(df_test, batch_size=Config.BATCH_SIZE)

    # Verification
    print(f"  Prediction shape: {submission.shape}")
    print(f"  Columns: {list(submission.columns)}")

    assert (
        "id" in submission.columns and "after" in submission.columns
    ), "Submission columns missing"
    assert len(submission) == len(df_test), "Submission length mismatch"

    # Check a few values
    print("  Sample Predictions:")
    print(submission.head())

    # Save submission explicitly for the demo check
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify output file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"  Submission file saved at: {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # 1. Set Seed for Reproducibility
    set_seed(42)

    # 2. Configure Environment (Paths & Hyperparams)
    configure_demo_environment()

    # 3. Create Mini Datasets (Overrides the paths in Config)
    train_path, val_path, test_path = create_mini_datasets(n_samples=200)
    Config.TRAIN_FILE = train_path
    Config.VAL_FILE = val_path
    Config.TEST_FILE = test_path

    # 4. Run Demos
    try:
        # Tier 1: HFBB
        demo_hfbb_engine()

        # Tokenizer
        tokenizer = demo_tokenizer()

        # Tier 2: Neural Training
        demo_training_pipeline(tokenizer)

        # Full Pipeline Inference
        demo_inference_pipeline()

        print("\n[Demo] All tests passed successfully!")

    except Exception as e:
        print(f"\n[Demo] FAILED with error: {e}")
        raise e
