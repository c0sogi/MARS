import os
import pandas as pd
import torch
import shutil
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, jaccard, clean_text
from library.data_processing import (
    get_qa_data,
    prepare_tapt_data,
    get_tokenizer,
    qa_collate_fn,
    QADataset,
    TAPTDataset,
)
from library.tapt_engine import run_tapt_training
from library.qa_engine import run_qa_training
from library.inference_engine import generate_submission


def setup_demo_environment():
    """
    Configures the environment for a fast demo run.
    Patches Config, creates directories, and subsets data.
    """
    print("Setting up demo environment...")

    # 1. Patch Configuration for Speed and Isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.QA_CACHE_DIR = os.path.join(Config.WORKING_DIR, "qa_cache")
    Config.QA_MODELS_DIR = os.path.join(Config.WORKING_DIR, "qa_models")
    Config.TAPT_CACHE_DIR = os.path.join(Config.WORKING_DIR, "tapt_cache")
    Config.TAPT_OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "tapt_model_finetuned")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Metadata paths to point to our subset location
    demo_meta_dir = os.path.join(Config.WORKING_DIR, "metadata")
    Config.TRAIN_META_PATH = os.path.join(demo_meta_dir, "train.csv")
    Config.VAL_META_PATH = os.path.join(demo_meta_dir, "val.csv")
    Config.TEST_META_PATH = os.path.join(demo_meta_dir, "test.csv")

    # Reduce computational load
    Config.EPOCHS = 1
    Config.TAPT_EPOCHS = 1
    Config.SEED_LIST = [42]  # Only run one seed
    Config.BATCH_SIZE = 4  # Small batch for demo
    Config.TAPT_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Initialize directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(demo_meta_dir, exist_ok=True)
    Config.setup()

    # 2. Create Data Subsets
    # We read from the original metadata provided in the environment
    # and save small chunks to our demo metadata folder.
    orig_meta_dir = "./metadata"

    # Subset Train
    df_train = pd.read_csv(os.path.join(orig_meta_dir, "train.csv"))
    df_train_sub = df_train.head(20)  # 20 samples
    df_train_sub.to_csv(Config.TRAIN_META_PATH, index=False)
    print(f"Created subset train data: {len(df_train_sub)} rows")

    # Subset Val
    df_val = pd.read_csv(os.path.join(orig_meta_dir, "val.csv"))
    df_val_sub = df_val.head(10)  # 10 samples
    df_val_sub.to_csv(Config.VAL_META_PATH, index=False)
    print(f"Created subset val data: {len(df_val_sub)} rows")

    # Subset Test
    df_test = pd.read_csv(os.path.join(orig_meta_dir, "test.csv"))
    df_test_sub = df_test.head(10)  # 10 samples
    df_test_sub.to_csv(Config.TEST_META_PATH, index=False)
    print(f"Created subset test data: {len(df_test_sub)} rows")


def test_utils():
    """Verifies utility functions."""
    print("\nTesting Utilities...")

    # Test Jaccard
    s1 = "This is a test answer"
    s2 = "this is test answer"
    score = jaccard(s1, s2)
    # intersection: {this, is, test, answer} (len 4)
    # union: {this, is, a, test, answer} (len 5)
    # score: 4/5 = 0.8
    assert (
        abs(score - 0.8) < 1e-6
    ), f"Jaccard calculation incorrect. Expected 0.8, got {score}"

    s3 = "Completely different"
    score_zero = jaccard(s1, s3)
    assert score_zero == 0.0, "Jaccard should be 0 for disjoint sets"

    # Test Clean Text
    raw_text = "  Sample   text\nwith  newlines "
    cleaned = clean_text(raw_text)
    assert (
        cleaned == "Sample text with newlines"
    ), f"Clean text failed. Got: '{cleaned}'"

    print("Utilities verified.")


def test_data_processing():
    """Verifies data loading and processing."""
    print("\nTesting Data Processing...")

    tokenizer = get_tokenizer()

    # Test TAPT Data Preparation
    tapt_ds = prepare_tapt_data(tokenizer)
    assert isinstance(
        tapt_ds, TAPTDataset
    ), "prepare_tapt_data should return TAPTDataset"
    assert len(tapt_ds) > 0, "TAPT dataset should not be empty"
    print(f"TAPT Dataset size (windows): {len(tapt_ds)}")

    # Test QA Data Loading
    # force reload to ensure it reads our subset files
    train_ds, val_ds, test_ds = get_qa_data(tokenizer, load_cached_data=False)

    assert isinstance(train_ds, QADataset)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    # Test Collate Function
    loader = DataLoader(train_ds, batch_size=2, collate_fn=qa_collate_fn)
    batch = next(iter(loader))

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].shape[0] == 2

    print("Data Processing verified.")


def test_tapt_engine():
    """Verifies TAPT training loop."""
    print("\nTesting TAPT Engine...")

    # Run training
    # force_retrain=True to ensure we actually run the code path
    output_dir = run_tapt_training(force_retrain=True)

    # Verify outputs
    assert os.path.exists(output_dir), "TAPT output directory not found"
    assert os.path.exists(
        os.path.join(output_dir, "model.safetensors")
    ) or os.path.exists(
        os.path.join(output_dir, "pytorch_model.bin")
    ), "Model weights not found"
    assert os.path.exists(
        os.path.join(output_dir, "tokenizer.json")
    ), "Tokenizer not saved"

    print("TAPT Engine verified.")
    return output_dir


def test_qa_engine(tapt_model_path):
    """Verifies QA training loop."""
    print("\nTesting QA Engine...")

    # Run QA training using the model we just fine-tuned
    run_qa_training(tapt_model_dir=tapt_model_path)

    # Verify model saved for the seed
    seed = Config.SEED_LIST[0]
    model_path = os.path.join(Config.QA_MODELS_DIR, f"model_seed_{seed}.pt")
    assert os.path.exists(model_path), f"QA Model for seed {seed} not found"

    print("QA Engine verified.")


def test_inference_engine():
    """Verifies Inference and Submission generation."""
    print("\nTesting Inference Engine...")

    # Generate submission
    generate_submission()

    # Check file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Validate content format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df.columns, "Submission missing 'id' column"
    assert (
        "PredictionString" in df.columns
    ), "Submission missing 'PredictionString' column"

    # Check if we have predictions for our subset test IDs
    test_meta = pd.read_csv(Config.TEST_META_PATH)
    expected_ids = set(test_meta["id"].unique())
    submitted_ids = set(df["id"].unique())

    assert expected_ids.issubset(submitted_ids), "Submission missing some test IDs"

    print(f"Submission generated with {len(df)} rows.")
    print("Inference Engine verified.")


def main():
    # Set global seed
    set_seed(42)

    # 1. Setup
    setup_demo_environment()

    # 2. Verify Utilities
    test_utils()

    # 3. Verify Data Processing
    test_data_processing()

    # 4. Run TAPT
    tapt_model_path = test_tapt_engine()

    # 5. Run QA Training
    test_qa_engine(tapt_model_path)

    # 6. Run Inference
    test_inference_engine()

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
