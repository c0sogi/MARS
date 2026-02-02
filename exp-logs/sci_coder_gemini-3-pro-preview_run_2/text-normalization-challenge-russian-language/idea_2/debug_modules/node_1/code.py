import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil

# Import library modules
from library.config import Config, set_seed
from library.vocab import CharTokenizer, get_tokenizer
from library.hfbb import HFBBModel
from library.dataset import DigitSeq2SeqDataset
from library.model import ContextAwareTransformer
from library.trainer import Trainer
from library.inference import HybridNormalizer


def create_mini_datasets():
    """
    Creates small subsets of the original data to speed up the demo.
    """
    print("Creating mini-datasets for rapid demonstration...")

    # Define source paths (from metadata)
    src_train = "./metadata/train.csv"
    src_val = "./metadata/val.csv"
    src_test = "./metadata/test.csv"

    # Define destination paths (in temp working dir)
    dst_train = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    dst_val = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    dst_test = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Read top 1000 rows and save
    # We ensure we get some digits for the Seq2Seq model
    df_train = pd.read_csv(src_train, nrows=2000)
    df_train.to_csv(dst_train, index=False)

    df_val = pd.read_csv(src_val, nrows=500)
    df_val.to_csv(dst_val, index=False)

    df_test = pd.read_csv(src_test, nrows=100)
    df_test.to_csv(dst_test, index=False)

    # Update Config paths to point to these mini datasets
    Config.TRAIN_DATA_PATH = dst_train
    Config.VAL_DATA_PATH = dst_val
    Config.TEST_DATA_PATH = dst_test

    print(f"Mini-datasets created at {Config.WORKING_DIR}")


def main():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    print("\n[1] Setting up Configuration...")

    # Set working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.setup_directories()

    # Update internal paths based on new working dir
    Config.HFBB_CACHE_DIR = os.path.join(Config.WORKING_DIR, "hfbb_cache")
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "seq2seq_demo.pth")
    Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.json")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Optimize hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 500
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.EMBED_DIM = 64
    Config.HIDDEN_DIM = 128
    Config.N_LAYERS = 2
    Config.N_HEADS = 2
    Config.DEVICE = "cpu"  # Force CPU for simple demo stability/speed on small data

    set_seed(42)

    # Prepare Data
    create_mini_datasets()

    # ==========================================
    # 2. Tokenizer Demonstration
    # ==========================================
    print("\n[2] Testing Tokenizer...")

    # Build tokenizer from the mini training data
    tokenizer = get_tokenizer(load_cached_data=False)

    # Test Encoding
    sample_text = "123"
    encoded = tokenizer.encode(sample_text, add_special_tokens=True)
    # Expect: SOS + 1 + 2 + 3 + EOS
    assert len(encoded) == len(sample_text) + 2, "Encoding length mismatch"
    assert encoded[0] == Config.SOS_IDX, "First token must be SOS"
    assert encoded[-1] == Config.EOS_IDX, "Last token must be EOS"

    # Test Decoding
    decoded = tokenizer.decode(encoded, remove_special_tokens=True)
    assert (
        decoded == sample_text
    ), f"Decoding failed. Expected '{sample_text}', got '{decoded}'"

    print(f"Tokenizer vocab size: {len(tokenizer.token2idx)}")
    print("Tokenizer verified.")

    # ==========================================
    # 3. HFBB Model Demonstration
    # ==========================================
    print("\n[3] Testing HFBB Model...")

    hfbb = HFBBModel()
    # Fit on the mini training data
    hfbb.fit(train_df=pd.read_csv(Config.TRAIN_DATA_PATH), load_cached_data=False)

    # Basic prediction test (Identity backoff)
    # "unknown_token" is unlikely to be in the top 2000 rows with a mapping
    res = hfbb.predict("unknown_token")
    assert res == "unknown_token", "HFBB should backoff to identity for unknown tokens"

    print("HFBB Model verified.")

    # ==========================================
    # 4. Dataset Demonstration
    # ==========================================
    print("\n[4] Testing Dataset...")

    # Initialize dataset (filters for digits)
    train_ds = DigitSeq2SeqDataset(
        mode="train", tokenizer=tokenizer, load_cached_data=False, debug=True
    )

    if len(train_ds) > 0:
        item = train_ds[0]
        # Verify keys
        required_keys = ["src", "tgt", "raw_before", "raw_after", "id"]
        for k in required_keys:
            assert k in item, f"Missing key {k} in dataset item"

        # Verify tensor shapes
        assert isinstance(item["src"], torch.Tensor)
        assert len(item["src"]) == Config.MAX_SEQ_LEN

        print(f"Dataset contains {len(train_ds)} samples with digits.")
    else:
        print(
            "Warning: No digit samples found in mini-dataset. Skipping dataset assertions."
        )

    print("Dataset verified.")

    # ==========================================
    # 5. Transformer Model Demonstration
    # ==========================================
    print("\n[5] Testing Transformer Architecture...")

    vocab_size = len(tokenizer.token2idx)
    model = ContextAwareTransformer(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        n_layers=Config.N_LAYERS,
        n_heads=Config.N_HEADS,
        device=Config.DEVICE,
    )

    # Dummy forward pass
    batch_size = 2
    seq_len = 10
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(Config.DEVICE)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len)).to(Config.DEVICE)

    output = model(src, tgt)

    # Expected output: [Batch, Seq, Vocab]
    assert output.shape == (
        batch_size,
        seq_len,
        vocab_size,
    ), f"Output shape mismatch: {output.shape}"
    print("Model forward pass verified.")

    # ==========================================
    # 6. Training Loop Demonstration
    # ==========================================
    print("\n[6] Testing Training Loop...")

    trainer = Trainer(tokenizer=tokenizer)

    # Create validation dataset
    val_ds = DigitSeq2SeqDataset(
        mode="val", tokenizer=tokenizer, load_cached_data=False, debug=True
    )

    # Run training (1 epoch as per config)
    # If dataset is empty (no digits in top 2000 rows), this might be trivial, but code should run.
    if len(train_ds) > 0:
        trained_model = trainer.fit(
            train_dataset=train_ds, val_dataset=val_ds, load_cached_data=False
        )

        # Verify checkpoint existence
        assert os.path.exists(
            Config.MODEL_CHECKPOINT
        ), "Model checkpoint was not saved."
        print("Training loop completed successfully.")
    else:
        print("Skipping training loop due to empty dataset.")

    # ==========================================
    # 7. Inference Demonstration
    # ==========================================
    print("\n[7] Testing Inference Pipeline...")

    # Load mini test set
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Initialize HybridNormalizer
    # This loads the HFBB model and the Seq2Seq model we just trained
    normalizer = HybridNormalizer(load_cached_data=True)

    # Run prediction
    result_df = normalizer.predict(test_df)

    # Verify results
    assert "after" in result_df.columns, "Result DataFrame missing 'after' column"
    assert len(result_df) == len(test_df), "Result length mismatch"

    # Check a sample
    print("Sample Predictions:")
    print(result_df[["before", "after"]].head(5))

    print("Inference pipeline verified.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
