import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
from unittest.mock import patch

# Import library modules
from library.config import Config
from library.utils import clean_text, calculate_f1_score, seed_everything
from library.data_processing import get_dataloaders, Tokenizer
from library.model import TextCNN
from library.engine import run_training


def main():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("--- 1. Configuring Environment for Demo ---")

    # Set fixed seed
    seed_everything(42)

    # Define a demo working directory to avoid messing with real experiment data
    DEMO_WORKING_DIR = "./working/demo_pipeline"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config parameters for speed and low memory usage
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")

    # Update cache paths to point to the demo directory
    Config.TRAIN_TOKENS_PATH = os.path.join(DEMO_WORKING_DIR, "train_tokens.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(DEMO_WORKING_DIR, "train_labels.npy")
    Config.VAL_TOKENS_PATH = os.path.join(DEMO_WORKING_DIR, "val_tokens.npy")
    Config.VAL_LABELS_PATH = os.path.join(DEMO_WORKING_DIR, "val_labels.npy")
    Config.TEST_TOKENS_PATH = os.path.join(DEMO_WORKING_DIR, "test_tokens.npy")
    Config.TEST_IDS_PATH = os.path.join(DEMO_WORKING_DIR, "test_ids.npy")
    Config.VOCAB_PATH = os.path.join(DEMO_WORKING_DIR, "vocab.json")
    Config.MLB_PATH = os.path.join(DEMO_WORKING_DIR, "mlb.joblib")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_WORKING_DIR, "model_demo.pth")

    # Reduce hyperparameters for fast execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.MAX_LEN = 50  # Short sequences
    Config.VOCAB_SIZE = 1000  # Small vocabulary
    Config.TOP_K_TAGS = 20  # Predict only top 20 tags
    Config.EMBED_DIM = 64  # Small embeddings
    Config.NUM_FILTERS = 16  # Few filters
    Config.KERNEL_SIZES = [3]  # Simple kernel
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    print("Configuration updated for demonstration.")

    # ==========================================
    # 2. Utility Validation
    # ==========================================
    print("\n--- 2. Validating Utilities ---")

    # Test clean_text
    raw_text = "<p>Hello World!</p>  Code: <code>print(x)</code>"
    cleaned = clean_text(raw_text)
    expected = "hello world code print x"
    print(f"Raw: '{raw_text}' -> Cleaned: '{cleaned}'")
    assert cleaned == expected, f"clean_text failed. Got {cleaned}"

    # Test f1_score
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[0, 1, 0], [1, 0, 0]])
    # Sample 1: True=[B, C], Pred=[B]. TP=1, FN=1, FP=0.
    # Sample 2: True=[A], Pred=[A]. TP=1, FN=0, FP=0.
    # Total: TP=2, FN=1, FP=0.
    # Precision = 2/2 = 1.0. Recall = 2/3 = 0.66. F1 = 2*(1*0.66)/(1+0.66) = 0.8
    f1 = calculate_f1_score(y_true, y_pred, average="micro")
    print(f"Calculated F1 Score: {f1:.4f}")
    assert 0.79 < f1 < 0.81, "F1 Score calculation incorrect"

    print("Utilities validated.")

    # ==========================================
    # 3. Data Processing with Mocking
    # ==========================================
    print("\n--- 3. Processing Data (Mocked) ---")

    # We patch pandas.read_csv to read only a subset of the raw data files
    # This prevents loading the 130M+ row dataset into memory.
    original_read_csv = pd.read_csv

    def mocked_read_csv(*args, **kwargs):
        # Determine the file path
        path = args[0] if args else kwargs.get("filepath_or_buffer")

        # Check if we are reading raw input files
        if (
            isinstance(path, str)
            and "input" in path
            and ("train.csv" in path or "test.csv" in path)
        ):
            print(
                f"[Mock] Intercepting read for {os.path.basename(path)}. Limiting to 2000 rows."
            )
            kwargs["nrows"] = 2000

        return original_read_csv(*args, **kwargs)

    # Apply the patch
    with patch("pandas.read_csv", side_effect=mocked_read_csv):
        # Force reload of data (ignore existing cache if any)
        train_loader, val_loader, test_loader, tokenizer, mlb = get_dataloaders(
            load_cached_data=False
        )

    print("DataLoaders created successfully.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    print(f"Number of classes (Tags): {len(mlb.classes_)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    tokens = batch["tokens"]
    labels = batch["labels"]

    assert tokens.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Token shape mismatch. Expected {(Config.BATCH_SIZE, Config.MAX_LEN)}, got {tokens.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        len(mlb.classes_),
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE, len(mlb.classes_))}, got {labels.shape}"

    print("Batch structure verified.")

    # ==========================================
    # 4. Model Verification
    # ==========================================
    print("\n--- 4. Verifying Model Architecture ---")

    device = torch.device("cpu")  # Use CPU for simple check
    model = TextCNN(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=len(mlb.classes_),
        kernel_sizes=Config.KERNEL_SIZES,
        num_filters=Config.NUM_FILTERS,
        dropout=Config.DROPOUT,
    ).to(device)

    # Forward pass check
    dummy_input = torch.randint(
        0, Config.VOCAB_SIZE, (Config.BATCH_SIZE, Config.MAX_LEN)
    ).to(device)
    output = model(dummy_input)

    assert output.shape == (
        Config.BATCH_SIZE,
        len(mlb.classes_),
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, len(mlb.classes_))}, got {output.shape}"

    print("Model forward pass successful.")

    # ==========================================
    # 5. Full Training Pipeline
    # ==========================================
    print("\n--- 5. Running Training Loop ---")

    # We run the training engine. This uses the mocked data loaded previously.
    # Note: run_training initializes a new model internally, so we pass the loaders we created.

    run_training(train_loader, val_loader, test_loader, mlb)

    # ==========================================
    # 6. Submission Verification
    # ==========================================
    print("\n--- 6. Verifying Submission ---")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(df_sub)}")
    print("First 3 rows:")
    print(df_sub.head(3))

    # Check if Id and Tags columns exist
    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission missing required columns"

    # Check that we have predictions for the test set (loaded via mock)
    # Note: Since we mocked the test set to 2000 rows, submission should have 2000 rows.
    # The actual test.csv has 600k rows, but our loader only saw 2000.
    assert (
        len(df_sub) == 2000
    ), f"Expected 2000 predictions (mocked limit), got {len(df_sub)}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
