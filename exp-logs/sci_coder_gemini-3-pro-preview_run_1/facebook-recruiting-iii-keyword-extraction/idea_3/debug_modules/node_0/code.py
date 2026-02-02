import os
import pandas as pd
import torch
import numpy as np
import warnings
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed
from library.vocabulary import Vocabulary, get_or_build_vocabulary
from library.dataset import StackExchangeDataset, get_loader
from library.model import DualStreamAttentionDAN
from library.engine import train_model, predict_test

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Self-Contained Demo ===")

    # ------------------------------------------------------------------------
    # 1. Setup Demo Environment & Data Subset
    # ------------------------------------------------------------------------
    # We create a separate directory structure in ./working to hold our mini-dataset
    # This ensures the demo runs fast (seconds instead of hours) and doesn't
    # overwrite the main run's artifacts.

    DEMO_BASE = "./working/demo_env"
    DEMO_INPUT = os.path.join(DEMO_BASE, "input")
    DEMO_META = os.path.join(DEMO_BASE, "metadata")
    DEMO_WORK = os.path.join(DEMO_BASE, "working")
    DEMO_SUB = os.path.join(DEMO_BASE, "submission")

    for d in [DEMO_INPUT, DEMO_META, DEMO_WORK, DEMO_SUB]:
        os.makedirs(d, exist_ok=True)

    print(f"Created demo environment at {DEMO_BASE}")

    # --- Create Mini Datasets ---
    # We read only the first 100 rows of the actual train.csv and 20 of test.csv
    # to simulate the dataset files.

    print("Creating mini-datasets from source files...")

    # Read subset of Train
    df_train_full = pd.read_csv("./input/train.csv", nrows=100)
    # Ensure no missing tags in our subset
    df_train_full.dropna(subset=["Tags"], inplace=True)

    # Save to demo input location
    demo_train_csv_path = os.path.join(DEMO_INPUT, "train.csv")
    df_train_full.to_csv(demo_train_csv_path, index=False)

    # Read subset of Test
    df_test_full = pd.read_csv("./input/test.csv", nrows=20)
    demo_test_csv_path = os.path.join(DEMO_INPUT, "test.csv")
    df_test_full.to_csv(demo_test_csv_path, index=False)

    # --- Generate Mini Metadata ---
    # Split the 100 train rows into 80 Train / 20 Val
    train_subset = df_train_full.iloc[:80].copy()
    val_subset = df_train_full.iloc[80:].copy()

    # Metadata only needs Id, Tags (for train/val), and file_path
    # Note: The library expects 'file_path' to be relative or handled by Config.
    # The library logic in dataset.py actually merges metadata with Config.TRAIN_CSV directly
    # based on ID. The metadata file itself just needs IDs and Tags.

    train_subset[["Id", "Tags"]].to_csv(
        os.path.join(DEMO_META, "train_metadata.csv"), index=False
    )
    val_subset[["Id", "Tags"]].to_csv(
        os.path.join(DEMO_META, "val_metadata.csv"), index=False
    )
    df_test_full[["Id"]].to_csv(
        os.path.join(DEMO_META, "test_metadata.csv"), index=False
    )

    print("Mini-datasets and metadata created successfully.")

    # ------------------------------------------------------------------------
    # 2. Override Configuration
    # ------------------------------------------------------------------------
    # Point the Config class to our demo directories and adjust params for speed.

    print("Overriding Config parameters for demo...")
    Config.INPUT_DIR = DEMO_INPUT
    Config.METADATA_DIR = DEMO_META
    Config.WORKING_DIR = DEMO_WORK
    Config.SUBMISSION_DIR = DEMO_SUB

    Config.TRAIN_CSV = demo_train_csv_path
    Config.TEST_CSV = demo_test_csv_path

    Config.TRAIN_META = os.path.join(DEMO_META, "train_metadata.csv")
    Config.VAL_META = os.path.join(DEMO_META, "val_metadata.csv")
    Config.TEST_META = os.path.join(DEMO_META, "test_metadata.csv")

    Config.MODEL_PATH = os.path.join(DEMO_WORK, "best_model.pth")
    Config.TOKENIZER_PATH = os.path.join(DEMO_WORK, "tokenizer.json")
    Config.TAG_MAP_PATH = os.path.join(DEMO_WORK, "tag_map.json")
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUB, "submission.csv")

    Config.CACHED_TRAIN = os.path.join(DEMO_WORK, "cached_train.parquet")
    Config.CACHED_VAL = os.path.join(DEMO_WORK, "cached_val.parquet")
    Config.CACHED_TEST = os.path.join(DEMO_WORK, "cached_test.parquet")

    # Hyperparameters for demo
    Config.VOCAB_SIZE = 1000  # Small vocab
    Config.BATCH_SIZE = 16
    Config.NUM_EPOCHS = 2
    Config.EMBED_DIM = 32
    Config.HIDDEN_DIM = 64

    # Re-run setup to ensure directories exist
    Config.setup()
    set_seed(Config.SEED)

    # ------------------------------------------------------------------------
    # 3. Vocabulary Building
    # ------------------------------------------------------------------------
    print("\n--- Testing Vocabulary ---")
    # Force build from scratch by setting load_cached_data=False
    # In a real run, we might rely on cache, but here we want to test the build logic.
    vocab = get_or_build_vocabulary(load_cached_data=False)

    # Validations
    print(f"Vocab Size: {vocab.get_vocab_size()}")
    print(f"Num Tags: {vocab.get_num_tags()}")

    assert (
        vocab.get_vocab_size() > 2
    ), "Vocabulary should contain more than just PAD and UNK."
    assert vocab.get_num_tags() > 0, "Tags should be detected from the training subset."

    # Test text conversion
    test_text = "python java code"
    indices = vocab.text_to_indices(test_text)
    assert len(indices) == 3, "Text to indices failed."
    assert isinstance(indices[0], int), "Indices must be integers."

    # Test tag conversion
    # Pick a tag that definitely exists in the top 100 rows (e.g., from the first row)
    first_tags = df_train_full.iloc[0]["Tags"]
    tag_indices = vocab.tags_to_indices(first_tags)
    reconstructed = vocab.indices_to_tags(tag_indices)

    # Note: reconstructed might be sorted differently or missing if tags were filtered (though we keep all tags)
    # Just check that we got indices back.
    assert len(tag_indices) > 0, "Tag conversion failed."

    # ------------------------------------------------------------------------
    # 4. Dataset & DataLoader
    # ------------------------------------------------------------------------
    print("\n--- Testing Dataset & DataLoader ---")

    # Create Train Loader
    train_loader = get_loader(
        "train", vocab, batch_size=Config.BATCH_SIZE, shuffle=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Validate Batch Structure
    required_keys = [
        "title_text",
        "title_offsets",
        "body_text",
        "body_offsets",
        "targets",
        "ids",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Validate Shapes
    # targets should be (Batch_Size, Num_Tags)
    assert batch["targets"].dim() == 2, "Targets should be 2D tensor."
    assert (
        batch["targets"].shape[1] == vocab.get_num_tags()
    ), "Targets dim 1 should match num tags."

    # Check offsets logic: Offsets should be monotonic increasing
    # title_offsets: [0, len1, len1+len2, ...]
    assert torch.all(
        batch["title_offsets"][1:] >= batch["title_offsets"][:-1]
    ), "Offsets must be monotonic."

    print("Batch validation passed.")

    # ------------------------------------------------------------------------
    # 5. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n--- Testing Model ---")

    model = DualStreamAttentionDAN(
        vocab_size=vocab.get_vocab_size(),
        num_classes=vocab.get_num_tags(),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    )
    model.to(Config.DEVICE)

    # Move batch to device
    t_text = batch["title_text"].to(Config.DEVICE)
    t_off = batch["title_offsets"].to(Config.DEVICE)
    b_text = batch["body_text"].to(Config.DEVICE)
    b_off = batch["body_offsets"].to(Config.DEVICE)

    # Forward
    logits = model(t_text, t_off, b_text, b_off)

    # Validate Output
    assert logits.shape == (
        t_off.size(0),
        vocab.get_num_tags(),
    ), f"Logit shape mismatch. Expected {(t_off.size(0), vocab.get_num_tags())}, got {logits.shape}"

    print("Model forward pass successful.")

    # ------------------------------------------------------------------------
    # 6. Training Loop
    # ------------------------------------------------------------------------
    print("\n--- Testing Training Loop ---")

    val_loader = get_loader("val", vocab, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run training (2 epochs on tiny data)
    train_model(
        model, train_loader, val_loader, Config.DEVICE, num_epochs=Config.NUM_EPOCHS
    )

    assert os.path.exists(Config.MODEL_PATH), "Best model checkpoint was not saved."
    print("Training simulation complete.")

    # ------------------------------------------------------------------------
    # 7. Inference
    # ------------------------------------------------------------------------
    print("\n--- Testing Inference ---")

    test_loader = get_loader("test", vocab, batch_size=Config.BATCH_SIZE, shuffle=False)

    predict_test(vocab, test_loader, Config.DEVICE)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_sub) == 20, f"Submission should have 20 rows, found {len(df_sub)}"
    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission columns missing."

    print("Inference simulation complete.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
