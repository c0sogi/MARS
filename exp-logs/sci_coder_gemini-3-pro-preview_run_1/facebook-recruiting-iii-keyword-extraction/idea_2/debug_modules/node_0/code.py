import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, calculate_f1_score
from library.tokenizer import TextProcessor
from library.dataset import StackExchangeDataset
from library.model import FastTextClassifier
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_data(base_dir):
    """
    Generates synthetic data for demonstration purposes.
    Creates a mini version of the dataset structure in a temporary directory.
    """
    input_dir = os.path.join(base_dir, "input")
    metadata_dir = os.path.join(base_dir, "metadata")
    working_dir = os.path.join(base_dir, "working")
    submission_dir = os.path.join(base_dir, "submission")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # 1. Create Mock train.csv
    # We generate 100 samples with alternating tags to simulate a multi-label problem
    train_data = {
        "Id": range(1, 101),
        "Title": [f"Title for question {i} about python java" for i in range(1, 101)],
        "Body": [
            f"Body content explaining the issue with code snippet {i}."
            for i in range(1, 101)
        ],
        "Tags": [
            "python java" if i % 2 == 0 else "c# javascript" for i in range(1, 101)
        ],
    }
    df_train = pd.DataFrame(train_data)
    train_csv_path = os.path.join(input_dir, "train.csv")
    df_train.to_csv(train_csv_path, index=False)

    # 2. Create Mock test.csv (20 samples)
    test_data = {
        "Id": range(1001, 1021),
        "Title": [f"Test question title {i}" for i in range(1001, 1021)],
        "Body": [f"Test body content {i}" for i in range(1001, 1021)],
    }
    df_test = pd.DataFrame(test_data)
    test_csv_path = os.path.join(input_dir, "test.csv")
    df_test.to_csv(test_csv_path, index=False)

    # 3. Create Metadata
    # Split 100 rows: 80 Train, 20 Validation
    df_train_meta = df_train.iloc[:80][["Id", "Tags"]].copy()
    df_train_meta["file_path"] = "train.csv"

    df_val_meta = df_train.iloc[80:][["Id", "Tags"]].copy()
    df_val_meta["file_path"] = "train.csv"

    df_test_meta = df_test[["Id"]].copy()
    df_test_meta["file_path"] = "test.csv"

    train_meta_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_meta_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_meta_path = os.path.join(metadata_dir, "test_metadata.csv")

    df_train_meta.to_csv(train_meta_path, index=False)
    df_val_meta.to_csv(val_meta_path, index=False)
    df_test_meta.to_csv(test_meta_path, index=False)

    return input_dir, metadata_dir, working_dir, submission_dir


def patch_config(input_dir, metadata_dir, working_dir, submission_dir):
    """
    Monkey-patches the Config class to use the demo directories and lighter hyperparameters.
    This ensures the code runs quickly on the synthetic data without modifying the library files.
    """
    # Override Paths
    Config.INPUT_DIR = input_dir
    Config.METADATA_DIR = metadata_dir
    Config.WORKING_DIR = working_dir
    Config.SUBMISSION_DIR = submission_dir

    Config.TRAIN_CSV = os.path.join(input_dir, "train.csv")
    Config.TEST_CSV = os.path.join(input_dir, "test.csv")
    Config.SAMPLE_SUBMISSION = os.path.join(input_dir, "sample_submission.csv")

    Config.TRAIN_METADATA = os.path.join(metadata_dir, "train_metadata.csv")
    Config.VAL_METADATA = os.path.join(metadata_dir, "val_metadata.csv")
    Config.TEST_METADATA = os.path.join(metadata_dir, "test_metadata.csv")

    Config.TOKENIZER_PATH = os.path.join(working_dir, "tokenizer.json")
    Config.LABEL_ENCODER_PATH = os.path.join(working_dir, "tag_map.json")
    Config.BEST_MODEL_PATH = os.path.join(working_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Override Hyperparameters for Speed
    Config.VOCAB_SIZE = 100  # Small vocab for demo
    Config.MAX_LEN = 20  # Short sequences
    Config.NUM_TAGS = 5  # We only have ~4 unique tags in mock data
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 2  # Few epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    print("Configuration patched for demo execution.")


def run_demo():
    # 1. Setup
    print("--- Setting up Demo Environment ---")
    base_dir = "./working/demo_env"
    # Clean up previous run if exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    input_dir, meta_dir, work_dir, sub_dir = setup_demo_data(base_dir)
    patch_config(input_dir, meta_dir, work_dir, sub_dir)
    set_seed(Config.SEED)

    # 2. Tokenizer Verification
    print("\n--- Testing TextProcessor (Tokenizer) ---")
    tokenizer = TextProcessor()
    # Force fit from scratch (ignore cache=False)
    tokenizer.fit(load_cached_data=False)

    # Validation
    vocab_size = tokenizer.get_vocab_size()
    num_tags = tokenizer.get_num_tags()
    print(f"Vocab Size: {vocab_size}, Num Tags: {num_tags}")

    assert vocab_size > 2, "Vocabulary should contain more than just PAD/UNK"
    assert num_tags > 0, "Tag map should not be empty"

    # Test Encoding
    dummy_title = "python code"
    dummy_body = "java error"
    encoded = tokenizer.encode_text([dummy_title], [dummy_body])
    assert len(encoded) == 1
    assert isinstance(encoded[0], list)
    print("Tokenizer encoding check passed.")

    # 3. Dataset & DataLoader Verification
    print("\n--- Testing Dataset and DataLoader ---")
    train_dataset = StackExchangeDataset(
        metadata_path=Config.TRAIN_METADATA,
        tokenizer=tokenizer,
        split_name="train",
        load_cached_data=False,  # Force reload
    )

    assert (
        len(train_dataset) == 80
    ), f"Expected 80 training samples, got {len(train_dataset)}"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=StackExchangeDataset.collate_fn,
    )

    # Fetch one batch to verify collate_fn
    text_indices, offsets, targets, ids = next(iter(train_loader))

    print(
        f"Batch Shapes - Text: {text_indices.shape}, Offsets: {offsets.shape}, Targets: {targets.shape}"
    )
    assert targets.shape[1] == num_tags, "Target dimension mismatch"
    assert offsets.shape[0] == Config.BATCH_SIZE, "Offsets size mismatch"
    print("Dataset and DataLoader check passed.")

    # 4. Model & Training Verification
    print("\n--- Testing Model and Training ---")

    # Create Validation Loader
    val_dataset = StackExchangeDataset(
        metadata_path=Config.VAL_METADATA,
        tokenizer=tokenizer,
        split_name="val",
        load_cached_data=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=StackExchangeDataset.collate_fn,
    )

    # Initialize Model
    model = FastTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_tags,
        embedding_dim=32,  # Small dim for demo
        dropout=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        device="cpu",  # Use CPU for small demo to avoid overhead/GPU memory issues
    )

    # Run Training Loop
    trainer.fit(epochs=Config.EPOCHS)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    print("Training loop completed successfully.")

    # 5. Inference / Submission Verification
    print("\n--- Testing Inference and Submission ---")

    test_dataset = StackExchangeDataset(
        metadata_path=Config.TEST_METADATA,
        tokenizer=tokenizer,
        split_name="test",
        load_cached_data=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=StackExchangeDataset.collate_fn,
    )

    # Generate Submission
    trainer.generate_submission(test_loader, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content and format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print("First 3 rows of submission:")
    print(df_sub.head(3))

    assert list(df_sub.columns) == ["Id", "Tags"], "Submission columns mismatch"
    assert len(df_sub) == 20, "Submission row count mismatch"

    print("\nAll demo checks passed successfully!")


if __name__ == "__main__":
    run_demo()
