import os
import sys
import pandas as pd
import torch
import numpy as np

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, ensure_dir
from library.data_processing import CharTokenizer, prepare_neural_dataset
from library.symbolic_model import NgramLookup
from library.training_engine import Trainer
from library.inference_engine import HybridPredictor


def create_mini_datasets(config, num_train=2000, num_val=200, num_test=200):
    """
    Creates small subsets of the original metadata to speed up the demo.
    """
    print(f"Creating mini datasets in {config.input_dir}...")

    # Original paths (from the default config, assuming standard layout)
    orig_train = "./metadata/train.csv"
    orig_val = "./metadata/val.csv"
    orig_test = "./metadata/test.csv"

    # Load and slice
    # We use on_bad_lines='skip' or similar if needed, but data is clean.
    # We strictly type columns to avoid pandas inference issues.
    dtype_dict = {"before": object, "after": object, "class": object}

    df_train = pd.read_csv(orig_train, dtype=dtype_dict, nrows=num_train)
    df_val = pd.read_csv(orig_val, dtype=dtype_dict, nrows=num_val)
    df_test = pd.read_csv(orig_test, dtype={"before": object}, nrows=num_test)

    # Save to the demo specific input directory (which is config.input_dir)
    # Note: config.input_dir in DemoConfig points to ./working/demo_metadata
    ensure_dir(config.train_file)

    df_train.to_csv(config.train_file, index=False)
    df_val.to_csv(config.val_file, index=False)
    df_test.to_csv(config.test_file, index=False)

    print("Mini datasets created.")


class DemoConfig(Config):
    """
    Configuration optimized for a quick demonstration run.
    """

    def __init__(self):
        super().__init__()

        # Override paths to use a demo working directory
        self.working_dir = "./working/demo_run"
        os.makedirs(self.working_dir, exist_ok=True)

        # We will create mini metadata files in a subdir of working
        self.input_dir = "./working/demo_metadata"
        os.makedirs(self.input_dir, exist_ok=True)

        self.train_file = os.path.join(self.input_dir, "train.csv")
        self.val_file = os.path.join(self.input_dir, "val.csv")
        self.test_file = os.path.join(self.input_dir, "test.csv")

        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # Override Model Hyperparameters for Speed
        self.embedding_dim = 64
        self.nhead = 2
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dim_feedforward = 128
        self.max_len = 64  # Short sequences for demo

        # Override Training Hyperparameters
        self.batch_size = 16
        self.num_epochs = 1  # Just one epoch to prove it runs
        self.num_workers = 0  # Avoid multiprocessing overhead for tiny data

        # Re-generate hash based on new params
        self.config_hash = self._generate_hash()

        # Update artifact paths with new hash
        self.tokenizer_path = os.path.join(
            self.working_dir, f"tokenizer_{self.config_hash}.json"
        )
        self.ngram_stats_path = os.path.join(
            self.working_dir, f"ngram_stats_{self.config_hash}.npy"
        )
        self.model_best_path = os.path.join(
            self.working_dir, f"neural_model_{self.config_hash}.pt"
        )
        self.train_seq_path = os.path.join(
            self.working_dir, f"train_seq_{self.config_hash}.parquet"
        )
        self.val_seq_path = os.path.join(
            self.working_dir, f"val_seq_{self.config_hash}.parquet"
        )
        self.test_seq_path = os.path.join(
            self.working_dir, f"test_seq_{self.config_hash}.parquet"
        )


def run_demo():
    # 1. Initialize Config
    config = DemoConfig()
    seed_everything(config.seed)
    print(f"Running demo with Config Hash: {config.config_hash}")

    # 2. Create Mini Datasets
    # We need to create these files before any library function tries to load them
    create_mini_datasets(config)

    # 3. Demonstrate Tokenizer & Data Processing
    print("\n--- Testing Data Processing ---")
    # We force re-computation by setting load_cached_data=False for the demo
    df_train_proc, tokenizer = prepare_neural_dataset(
        config, split="train", load_cached_data=False
    )

    print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size > len(
        config.special_tokens
    ), "Tokenizer failed to learn vocabulary."

    # Test Encoding/Decoding
    sample_text = "test"
    ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)
    print(f"Original: '{sample_text}' -> IDs: {ids} -> Decoded: '{decoded}'")
    assert decoded == sample_text, "Tokenizer encode/decode cycle failed."

    # 4. Demonstrate Symbolic Model
    print("\n--- Testing Symbolic Model (N-gram) ---")
    ngram_model = NgramLookup(config)
    ngram_model.fit(load_cached_data=False)

    # Verify it learned something from the mini train set
    # We pick a token that definitely exists in the first 2000 rows.
    # Usually "PLAIN" tokens map to themselves.
    # Let's check the internal dictionary directly or use a known common word.
    # If the mini set is too small/random, we might miss specific words,
    # but let's check if the dicts are populated.
    print(f"Unigrams learned: {len(ngram_model.unigrams)}")
    assert len(ngram_model.unigrams) > 0, "N-gram model failed to learn unigrams."

    # 5. Demonstrate Neural Training
    print("\n--- Testing Neural Training ---")
    trainer = Trainer(config)
    # We pass load_cached_data=True here because we just computed the train seqs in step 3
    # and prepare_neural_dataset saved them to parquet.
    normalizer = trainer.run(load_cached_data=True)

    # Check if model file exists
    assert os.path.exists(config.model_best_path), "Model checkpoint was not saved."
    print("Neural training completed and model saved.")

    # 6. Demonstrate Hybrid Inference
    print("\n--- Testing Hybrid Inference ---")
    predictor = HybridPredictor(config)
    predictor.predict(load_cached_data=True)

    # Verify Submission
    assert os.path.exists(config.submission_path), "Submission file was not created."

    df_sub = pd.read_csv(config.submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print("Submission Head:")
    print(df_sub.head())

    # Check columns
    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission missing required columns."

    # Check that we have predictions matching the mini test set
    # Note: HybridPredictor groups by sentence, so the number of rows should match
    # the total tokens in the test set.
    df_test_raw = pd.read_csv(config.test_file)
    expected_count = len(df_test_raw)
    actual_count = len(df_sub)

    print(f"Expected Predictions: {expected_count}, Actual: {actual_count}")
    assert (
        actual_count == expected_count
    ), f"Mismatch in prediction count. Expected {expected_count}, got {actual_count}."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
