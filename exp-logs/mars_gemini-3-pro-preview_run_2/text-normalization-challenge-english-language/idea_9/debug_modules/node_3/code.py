import os
import pandas as pd
import torch
import shutil
import logging
import sys
import warnings

# Suppress warnings and verbose logs for clean output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.normalization_rules import apply_rule, NumberConverter
from library.trainer_router import train_router
from library.trainer_generator import train_generator
from library.inference_pipeline import predict_all


def setup_demo_environment():
    """
    Sets up a temporary directory structure and creates dummy data
    to demonstrate the pipeline without processing the full 7M+ row dataset.
    """
    print(">>> Setting up demo environment...")

    # Define demo paths
    demo_root = "./working/demo_execution"
    if os.path.exists(demo_root):
        shutil.rmtree(demo_root)
    os.makedirs(demo_root)

    demo_metadata = os.path.join(demo_root, "metadata")
    demo_cache = os.path.join(demo_root, "cache")
    demo_checkpoints = os.path.join(demo_root, "checkpoints")
    demo_submission = os.path.join(demo_root, "submission")

    for d in [demo_metadata, demo_cache, demo_checkpoints, demo_submission]:
        os.makedirs(d, exist_ok=True)

    # Override Config paths dynamically
    Config.WORKING_DIR = demo_root
    Config.METADATA_DIR = demo_metadata
    Config.CACHE_DIR = demo_cache
    Config.CHECKPOINT_DIR = demo_checkpoints
    Config.SUBMISSION_DIR = demo_submission

    # IMPORTANT: We must manually update the file paths because they were
    # initialized at module import time in Config.
    Config.TRAIN_FILE = os.path.join(demo_metadata, "train.csv")
    Config.VAL_FILE = os.path.join(demo_metadata, "val.csv")
    Config.TEST_FILE = os.path.join(demo_metadata, "test.csv")
    Config.SUBMISSION_FILE = os.path.join(demo_submission, "submission.csv")

    # Reduce hyperparameters for speed
    Config.ROUTER_EPOCHS = 1
    Config.GENERATOR_EPOCHS = 1
    Config.ROUTER_BATCH_SIZE = 2
    Config.GENERATOR_BATCH_SIZE = 2
    # Use smaller models if possible, but we stick to config defaults (DeBERTa/ByT5)
    # as we don't want to download new models, just use what's cached/available.
    # Note: On a fresh run, this might trigger downloads.

    # Create Dummy Train Data
    # Sentence 1: "I have 2 apples."
    # Sentence 2: "Call 911 now." (Path B example)
    train_data = [
        {
            "sentence_id": 1,
            "token_id": 0,
            "class": "PLAIN",
            "before": "I",
            "after": "I",
        },
        {
            "sentence_id": 1,
            "token_id": 1,
            "class": "PLAIN",
            "before": "have",
            "after": "have",
        },
        {
            "sentence_id": 1,
            "token_id": 2,
            "class": "CARDINAL",
            "before": "2",
            "after": "two",
        },
        {
            "sentence_id": 1,
            "token_id": 3,
            "class": "PLAIN",
            "before": "apples",
            "after": "apples",
        },
        {
            "sentence_id": 1,
            "token_id": 4,
            "class": "PUNCT",
            "before": ".",
            "after": ".",
        },
        {
            "sentence_id": 2,
            "token_id": 0,
            "class": "PLAIN",
            "before": "Call",
            "after": "Call",
        },
        {
            "sentence_id": 2,
            "token_id": 1,
            "class": "TELEPHONE",
            "before": "911",
            "after": "nine one one",
        },
        {
            "sentence_id": 2,
            "token_id": 2,
            "class": "PLAIN",
            "before": "now",
            "after": "now",
        },
        {
            "sentence_id": 2,
            "token_id": 3,
            "class": "PUNCT",
            "before": ".",
            "after": ".",
        },
    ]

    # Create Dummy Val Data
    val_data = [
        {
            "sentence_id": 3,
            "token_id": 0,
            "class": "PLAIN",
            "before": "It",
            "after": "It",
        },
        {
            "sentence_id": 3,
            "token_id": 1,
            "class": "PLAIN",
            "before": "is",
            "after": "is",
        },
        {
            "sentence_id": 3,
            "token_id": 2,
            "class": "DATE",
            "before": "2023",
            "after": "twenty twenty three",
        },
    ]

    # Create Dummy Test Data
    # Sentence 4: "Room 101."
    test_data = [
        {"sentence_id": 4, "token_id": 0, "before": "Room"},
        {"sentence_id": 4, "token_id": 1, "before": "101"},
        {"sentence_id": 4, "token_id": 2, "before": "."},
    ]

    # Save to CSV
    def save_csv(data, path, is_test=False):
        df = pd.DataFrame(data)
        # Create 'id' column
        df["id"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)
        df.to_csv(path, index=False)

    save_csv(train_data, Config.TRAIN_FILE)
    save_csv(val_data, Config.VAL_FILE)
    save_csv(test_data, Config.TEST_FILE, is_test=True)

    print(">>> Demo environment ready.")


def verify_normalization_rules():
    """
    Tests the deterministic rule logic.
    """
    print("\n>>> Verifying normalization rules...")

    # Test Integer conversion
    assert NumberConverter.integer_to_words(123) == "one hundred twenty three"
    assert NumberConverter.integer_to_words(0) == "zero"

    # Test Ordinal conversion
    assert NumberConverter.integer_to_ordinal(1) == "first"
    assert NumberConverter.integer_to_ordinal(23) == "twenty third"

    # Test Class Handlers via apply_rule
    # CARDINAL
    assert apply_rule("1,000", "CARDINAL") == "one thousand"
    # DIGIT
    assert apply_rule("07", "DIGIT") == "zero seven"
    # LETTERS
    assert apply_rule("U.S.A.", "LETTERS") == "u s a"
    # PLAIN
    assert apply_rule("Hello", "PLAIN") == "Hello"

    print(">>> Rules verification passed.")


def run_router_training():
    """
    Runs the router training loop on dummy data.
    """
    print("\n>>> Starting Router Training (Demo)...")
    # Force reload of data to ensure it picks up dummy files
    # We delete cache explicitly just in case
    cache_file = os.path.join(Config.CACHE_DIR, "router_train_processed.parquet")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    accuracy = train_router(
        epochs=Config.ROUTER_EPOCHS,
        batch_size=Config.ROUTER_BATCH_SIZE,
        load_cached_data=False,  # Force processing of dummy CSV
    )
    print(f">>> Router Training Complete. Accuracy: {accuracy:.4f}")

    # Verify checkpoint exists
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "router_best")
    if not os.path.exists(checkpoint_path):
        raise AssertionError("Router checkpoint was not saved.")


def run_generator_training():
    """
    Runs the generator training loop on dummy data.
    """
    print("\n>>> Starting Generator Training (Demo)...")

    # Force reload
    cache_file = os.path.join(Config.CACHE_DIR, "generator_train_processed.parquet")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    loss = train_generator(
        epochs=Config.GENERATOR_EPOCHS,
        batch_size=Config.GENERATOR_BATCH_SIZE,
        load_cached_data=False,
    )
    print(f">>> Generator Training Complete. Best Val Loss: {loss:.4f}")

    # Verify checkpoint exists
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "generator_best")
    if not os.path.exists(checkpoint_path):
        raise AssertionError("Generator checkpoint was not saved.")


def run_inference():
    """
    Runs the inference pipeline on dummy test data.
    """
    print("\n>>> Running Inference Pipeline...")

    # Force reload test cache
    cache_file = os.path.join(Config.CACHE_DIR, "router_test_processed.parquet")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    predict_all(load_cached_data=False, debug=False)

    # Verify submission
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise AssertionError("Submission file was not created.")

    df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f">>> Submission created with {len(df)} rows.")
    print(df.head())

    # Basic check: ensure IDs match test set
    test_df = pd.read_csv(Config.TEST_FILE)
    assert len(df) == len(test_df), "Submission row count mismatch."
    assert "id" in df.columns and "after" in df.columns, "Submission columns incorrect."


if __name__ == "__main__":
    # 1. Set Seed
    seed_everything(42)

    # 2. Setup Environment
    setup_demo_environment()

    # 3. Verify Rules
    verify_normalization_rules()

    # 4. Train Router
    # Note: This will download 'microsoft/deberta-v3-base' if not cached.
    # In the provided environment, packages are installed, but models might need download.
    # We assume internet access or cached models are available.
    try:
        run_router_training()
    except Exception as e:
        print(
            f"Router training failed (likely due to resource/download limits in demo): {e}"
        )
        # Create a fake checkpoint so pipeline can continue for demonstration
        os.makedirs(os.path.join(Config.CHECKPOINT_DIR, "router_best"), exist_ok=True)
        # We can't easily fake a model save without a model, so we might stop here if it fails.
        raise e

    # 5. Train Generator
    try:
        run_generator_training()
    except Exception as e:
        print(f"Generator training failed: {e}")
        raise e

    # 6. Inference
    run_inference()

    print("\n>>> Demo Execution Successfully Completed.")
