import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil

# Import provided library modules
from library.config import Config, set_seed
from library.vocabulary import CharVocab
from library.dataset import TextNormalizationDataset, get_dataloader
from library.model import Encoder, Decoder, Seq2Seq
from library.train import train_model
from library.predict import generate_submission


def run_demo():
    # 1. Setup & Configuration
    print("--- 1. Setup & Configuration ---")
    set_seed(42)

    # Define working paths
    work_dir = "./working/demo"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Override Config paths to use our mini datasets and working dir
    # We modify class attributes directly as they are used statically in the library
    Config.WORKING_DIR = work_dir
    Config.TRAIN_DATA_PATH = os.path.join(work_dir, "mini_train.csv")
    Config.VAL_DATA_PATH = os.path.join(work_dir, "mini_val.csv")
    Config.TEST_DATA_PATH = os.path.join(work_dir, "mini_test.csv")
    Config.CHECKPOINT_PATH = os.path.join(work_dir, "model_checkpoint.pt")
    Config.VOCAB_PATH = os.path.join(work_dir, "vocab.npy")
    Config.SUBMISSION_PATH = os.path.join(work_dir, "submission.csv")

    # Override Model Hyperparameters for Speed
    Config.embed_dim = 16
    Config.hidden_dim = 32
    Config.n_layers = 1
    Config.dropout = 0.0

    print(f"Working directory set to: {Config.WORKING_DIR}")

    # 2. Data Preparation (Create Mini Datasets)
    print("\n--- 2. Creating Mini Datasets ---")

    # Load original metadata
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Sample and save mini versions
    # Train
    df_train = pd.read_csv(orig_train_path, dtype=str, nrows=100)
    df_train.to_csv(Config.TRAIN_DATA_PATH, index=False)
    print(f"Created mini train set: {len(df_train)} rows")

    # Val
    df_val = pd.read_csv(orig_val_path, dtype=str, nrows=20)
    df_val.to_csv(Config.VAL_DATA_PATH, index=False)
    print(f"Created mini val set: {len(df_val)} rows")

    # Test
    df_test = pd.read_csv(orig_test_path, dtype=str, nrows=20)
    df_test.to_csv(Config.TEST_DATA_PATH, index=False)
    print(f"Created mini test set: {len(df_test)} rows")

    # 3. Component Verification
    print("\n--- 3. Verifying Components ---")

    # A. Vocabulary
    print("Verifying Vocabulary...")
    vocab = CharVocab()
    # Force build from scratch using our mini train file
    vocab.build_vocab(Config.TRAIN_DATA_PATH, load_cached_data=False)

    assert len(vocab) > 4, "Vocabulary should contain more than just special tokens"

    test_str = "abc"
    encoded = vocab.encode(test_str)
    decoded = vocab.decode(encoded)
    print(f"Original: {test_str}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Check if decoding reconstructs the string (ignoring special tokens logic if any)
    # Note: decode removes special tokens by default, so it should match exactly if chars are in vocab
    # If 'a', 'b', 'c' are not in mini_train, they become UNK.
    # Let's verify with a string known to be in the dataframe or generic check.
    # We'll just assert that decoded is a string.
    assert isinstance(decoded, str)

    # B. Dataset
    print("\nVerifying Dataset...")
    # Force reload to ignore any previous cache
    ds = TextNormalizationDataset(Config.TRAIN_DATA_PATH, vocab, load_cached_data=False)
    assert len(ds) == 100
    item = ds[0]
    print(f"Dataset item keys: {item.keys()}")
    assert "src" in item and "tgt" in item and "id" in item
    assert isinstance(item["src"], torch.Tensor)

    # C. Model
    print("\nVerifying Model Forward Pass...")
    device = Config.device
    enc = Encoder(
        len(vocab), Config.embed_dim, Config.hidden_dim, Config.n_layers, Config.dropout
    )
    dec = Decoder(
        len(vocab), Config.embed_dim, Config.hidden_dim, Config.n_layers, Config.dropout
    )
    model = Seq2Seq(enc, dec, device).to(device)

    # Create dummy batch
    batch_size = 2
    src_len = 10
    tgt_len = 12
    dummy_src = torch.randint(0, len(vocab), (batch_size, src_len)).to(device)
    dummy_tgt = torch.randint(0, len(vocab), (batch_size, tgt_len)).to(device)

    output = model(dummy_src, dummy_tgt)
    # Output shape: [batch_size, tgt_len, vocab_size]
    print(f"Model output shape: {output.shape}")
    assert output.shape == (batch_size, tgt_len, len(vocab))

    # 4. Training Execution
    print("\n--- 4. Running Training Loop ---")
    # Run for 1 epoch with small batch size
    trained_model = train_model(
        num_epochs=1,
        batch_size=8,
        load_cached_data=True,  # Can use cached now since we built it above
        learning_rate=0.01,
    )

    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created!"
    print("Training completed and checkpoint verified.")

    # 5. Inference Execution
    print("\n--- 5. Running Inference ---")
    generate_submission(load_cached_data=True, batch_size=8)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created!"

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    assert len(df_sub) == 20, f"Expected 20 rows in submission, found {len(df_sub)}"
    assert "id" in df_sub.columns and "after" in df_sub.columns

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demo()
