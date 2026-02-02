import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything, get_artifact_path
from library.data_processing import (
    load_and_group_data,
    get_tokenizer,
    NeuralDataset,
    CharTokenizer,
)
from library.symbolic_model import SymbolicLookup
from library.neural_model import CharSeq2SeqTransformer, NeuralTrainer
from library.trainer import ModelTrainer
from library.inference_engine import HybridRouter, generate_submission_file


def create_dummy_data(base_dir):
    """Creates small dummy CSV files for training, validation, and testing."""
    os.makedirs(base_dir, exist_ok=True)

    # Create Train Data
    # Sentence 0: "I have 2 cats" -> "I have two cats"
    # Sentence 1: "It is 10 pm" -> "It is ten pm"
    train_data = [
        {
            "sentence_id": 0,
            "token_id": 0,
            "class": "PLAIN",
            "before": "I",
            "after": "I",
        },
        {
            "sentence_id": 0,
            "token_id": 1,
            "class": "PLAIN",
            "before": "have",
            "after": "have",
        },
        {
            "sentence_id": 0,
            "token_id": 2,
            "class": "CARDINAL",
            "before": "2",
            "after": "two",
        },
        {
            "sentence_id": 0,
            "token_id": 3,
            "class": "PLAIN",
            "before": "cats",
            "after": "cats",
        },
        {
            "sentence_id": 1,
            "token_id": 0,
            "class": "PLAIN",
            "before": "It",
            "after": "It",
        },
        {
            "sentence_id": 1,
            "token_id": 1,
            "class": "PLAIN",
            "before": "is",
            "after": "is",
        },
        {
            "sentence_id": 1,
            "token_id": 2,
            "class": "CARDINAL",
            "before": "10",
            "after": "ten",
        },
        {
            "sentence_id": 1,
            "token_id": 3,
            "class": "PLAIN",
            "before": "pm",
            "after": "pm",
        },
    ]
    pd.DataFrame(train_data).to_csv(os.path.join(base_dir, "train.csv"), index=False)

    # Create Val Data (Similar structure)
    val_data = [
        {
            "sentence_id": 10,
            "token_id": 0,
            "class": "CARDINAL",
            "before": "5",
            "after": "five",
        },
    ]
    pd.DataFrame(val_data).to_csv(os.path.join(base_dir, "val.csv"), index=False)

    # Create Test Data (No 'after' or 'class')
    test_data = [
        {"sentence_id": 100, "token_id": 0, "before": "I"},
        {"sentence_id": 100, "token_id": 1, "before": "see"},
        {"sentence_id": 100, "token_id": 2, "before": "2"},
        {"sentence_id": 100, "token_id": 3, "before": "dogs"},
    ]
    pd.DataFrame(test_data).to_csv(os.path.join(base_dir, "test.csv"), index=False)

    return (
        os.path.join(base_dir, "train.csv"),
        os.path.join(base_dir, "val.csv"),
        os.path.join(base_dir, "test.csv"),
    )


def run_demo():
    # --- 1. Setup & Configuration Override ---
    print("--- 1. Setting up Demo Environment ---")

    # Define working directories
    work_dir = "./working"
    demo_meta_dir = os.path.join(work_dir, "demo_metadata")
    demo_artifact_dir = os.path.join(work_dir, "demo_artifacts")

    # Generate dummy data
    train_path, val_path, test_path = create_dummy_data(demo_meta_dir)

    # Override Config parameters to use dummy data and run fast
    Config.IDEA_DIR = demo_artifact_dir
    Config.TRAIN_META_PATH = train_path
    Config.VAL_META_PATH = val_path
    Config.TEST_META_PATH = test_path
    Config.EPOCHS = 1  # 1 Epoch for speed
    Config.BATCH_SIZE = 2  # Small batch size
    Config.VOCAB_SIZE = 50  # Small vocab
    Config.D_MODEL = 32  # Tiny model
    Config.NHEAD = 2
    Config.NUM_ENCODER_LAYERS = 1
    Config.NUM_DECODER_LAYERS = 1
    Config.DIM_FEEDFORWARD = 64

    seed_everything(Config.SEED)
    print("Configuration updated for demo run.")

    # --- 2. Data Processing Demo ---
    print("\n--- 2. Demonstrating Data Processing ---")

    # Load and Group Data
    print("Loading grouped data...")
    train_df = load_and_group_data("train", load_cached_data=False)
    assert len(train_df) == 2, f"Expected 2 sentences in train, got {len(train_df)}"
    assert isinstance(
        train_df.iloc[0]["before"], list
    ), "Grouped data should have lists of tokens"

    # Tokenizer
    print("Fitting tokenizer...")
    tokenizer = get_tokenizer(train_grouped_df=train_df, load_cached=False)
    encoded = tokenizer.encode("cat")
    decoded = tokenizer.decode(encoded)
    print(f"Tokenizer check: 'cat' -> {encoded} -> '{decoded}'")
    assert decoded == "cat", "Tokenizer decode mismatch"

    # Neural Dataset
    print("Creating NeuralDataset...")
    # Force sample_ratio=1.0 to ensure we get samples even from PLAIN text for this demo
    dataset = NeuralDataset(train_df, tokenizer, mode="train", sample_ratio=1.0)
    print(f"Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty"

    sample = dataset[0]
    assert "input_ids" in sample and "target_ids" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    print("Dataset item keys verified.")

    # --- 3. Symbolic Model Demo ---
    print("\n--- 3. Demonstrating Symbolic Model ---")

    # Initialize Symbolic Lookup (will build from train_df since cache is empty/invalidated)
    symbolic_model = SymbolicLookup(train_grouped_df=train_df, load_cached=False)

    # Test memorization: "2" -> "two"
    # In the dummy data, "2" appears as CARDINAL.
    # Context in sentence 0: "have" (prev), "2" (curr), "cats" (next)
    res_trigram = symbolic_model.query("have", "2", "cats")
    print(f"Symbolic Trigram Query ('have', '2', 'cats') -> '{res_trigram}'")
    assert (
        res_trigram == "two"
    ), f"Symbolic model failed to retrieve 'two', got {res_trigram}"

    # Test Unigram fallback: "10" -> "ten"
    res_unigram = symbolic_model.query("unknown", "10", "unknown")
    print(f"Symbolic Unigram Query ('unknown', '10', 'unknown') -> '{res_unigram}'")
    assert (
        res_unigram == "ten"
    ), f"Symbolic model failed to retrieve 'ten', got {res_unigram}"

    # --- 4. Neural Model Training Demo ---
    print("\n--- 4. Demonstrating Neural Training ---")

    trainer = ModelTrainer()
    # Force retrain to ensure we run the training loop
    best_model_path = trainer.run(force_retrain=True)

    assert os.path.exists(best_model_path), "Model checkpoint file was not created."
    print(f"Model successfully trained and saved to {best_model_path}")

    # --- 5. Inference Engine Demo ---
    print("\n--- 5. Demonstrating Inference Engine ---")

    # Initialize Router (loads symbolic stats and neural model)
    router = HybridRouter()

    # Generate submission for dummy test set
    submission_path = os.path.join(work_dir, "demo_submission.csv")
    router.generate_submission(output_file=submission_path)

    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(sub_df.head())

    # Check if we have predictions for all tokens
    # Test data has 4 tokens.
    assert len(sub_df) == 4, f"Expected 4 rows in submission, got {len(sub_df)}"

    # Check specific prediction logic
    # "2" in test set (row index 2) should be normalized.
    # Depending on whether symbolic or neural picked it up.
    # Symbolic unigram for "2" is "two".
    pred_2 = sub_df.iloc[2]["after"]
    print(f"Prediction for '2' in test set: '{pred_2}'")

    # Note: Since the neural model trained on 8 samples for 1 epoch, it might be garbage,
    # but the symbolic model should handle "2" if the context allows or via unigram backoff.
    # Our dummy test context for "2" is "see" (prev), "dogs" (next).
    # Trigram ("see", "2", "dogs") is unseen.
    # Bigram ("see", "2") is unseen.
    # Unigram "2" -> "two" is seen in training.
    # So SymbolicLookup should return "two".

    assert pred_2 == "two", f"Expected 'two', got '{pred_2}'"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
