import os
import torch
import pandas as pd
import numpy as np
import shutil
import sys

# Import library components
from library.config import Config
from library.data_utils import build_vocab
from library.models import SiameseRanker, SeparableConvReader
from library.dataset import (
    NQRankerDataset,
    ranker_collate_fn,
    NQReaderDataset,
    reader_collate_fn,
)
from library.trainer import RankerTrainer, ReaderTrainer
from library.inference import QuestionAnsweringPredictor


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DemoConfig(Config):
    """
    Configuration for the demo execution.
    Overrides paths to use a separate working directory and reduces
    dataset sizes/epochs for speed.
    """

    def __init__(self):
        # Initialize parent defaults
        super().__init__()

        # Override Working Directory
        self.WORKING_DIR = "./working/demo_execution/"
        self.SUBMISSION_DIR = "./working/demo_execution/submission/"

        # Ensure directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Update Cache Paths to point to demo directory
        self.VOCAB_CACHE_PATH = os.path.join(self.WORKING_DIR, "vocab.parquet")
        self.RANKER_TRAIN_CACHE = os.path.join(
            self.WORKING_DIR, "ranker_train_data.parquet"
        )
        self.RANKER_VAL_CACHE = os.path.join(
            self.WORKING_DIR, "ranker_val_data.parquet"
        )
        self.READER_TRAIN_CACHE = os.path.join(
            self.WORKING_DIR, "reader_train_data.parquet"
        )
        self.READER_VAL_CACHE = os.path.join(
            self.WORKING_DIR, "reader_val_data.parquet"
        )

        # Update Model Paths
        self.RANKER_MODEL_PATH = os.path.join(self.WORKING_DIR, "ranker_best.pth")
        self.READER_MODEL_PATH = os.path.join(self.WORKING_DIR, "reader_best.pth")

        # Update Submission Path
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # Speed Optimization: Reduce Data Size and Training Steps
        self.DEBUG_SAMPLE_SIZE = 50  # Only process 50 examples
        self.MAX_VOCAB_SIZE = 1000  # Smaller vocab for demo
        self.BATCH_SIZE = 4
        self.EPOCHS = 1
        self.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

        # Model dimensions (keep small for demo if needed, but defaults are fine)
        # We keep defaults to verify actual model architecture logic


def test_vocab_generation(config):
    print("\n=== Testing Vocabulary Generation ===")
    # Remove existing vocab if any to force rebuild
    if os.path.exists(config.VOCAB_CACHE_PATH):
        os.remove(config.VOCAB_CACHE_PATH)

    tokenizer = build_vocab(config, load_cached_data=False)

    # Validation
    assert os.path.exists(config.VOCAB_CACHE_PATH), "Vocab file was not created."
    assert (
        len(tokenizer.vocab) > 2
    ), "Vocabulary should contain more than just special tokens."
    print(f"Vocabulary generated successfully. Size: {len(tokenizer.vocab)}")
    return tokenizer


def test_ranker_model_logic(config, tokenizer):
    print("\n=== Testing SiameseRanker Model Logic ===")
    model = SiameseRanker(config)
    model.eval()

    # Create dummy input: Batch=2, SeqLen=10
    q_ids = torch.randint(0, len(tokenizer.vocab), (2, 10))
    cand_ids = torch.randint(0, len(tokenizer.vocab), (2, 20))

    # Forward pass
    scores = model(q_ids, cand_ids)

    # Validation
    assert scores.shape == (2,), f"Expected output shape (2,), got {scores.shape}"
    print("Ranker forward pass successful.")


def test_ranker_training(config):
    print("\n=== Testing Ranker Training Loop ===")
    trainer = RankerTrainer(config)

    # Train for 1 epoch on small subset
    # load_cached_data=False forces regeneration of the small subset defined in DemoConfig
    trainer.train(load_cached_data=False)

    # Validation
    assert os.path.exists(
        config.RANKER_MODEL_PATH
    ), "Ranker model checkpoint not found after training."
    print("Ranker training completed and model saved.")


def test_reader_model_logic(config, tokenizer):
    print("\n=== Testing SeparableConvReader Model Logic ===")
    model = SeparableConvReader(config)
    model.eval()

    # Create dummy input: Batch=2, SeqLen=50
    input_ids = torch.randint(0, len(tokenizer.vocab), (2, 50))

    # Forward pass
    start_logits, end_logits = model(input_ids)

    # Validation
    # Output should be (Batch, SeqLen) for both start and end logits
    assert start_logits.shape == (
        2,
        50,
    ), f"Expected start logits shape (2, 50), got {start_logits.shape}"
    assert end_logits.shape == (
        2,
        50,
    ), f"Expected end logits shape (2, 50), got {end_logits.shape}"
    print("Reader forward pass successful.")


def test_reader_training(config):
    print("\n=== Testing Reader Training Loop ===")
    trainer = ReaderTrainer(config)

    # Train for 1 epoch on small subset
    trainer.train(load_cached_data=False)

    # Validation
    assert os.path.exists(
        config.READER_MODEL_PATH
    ), "Reader model checkpoint not found after training."
    print("Reader training completed and model saved.")


def test_inference_pipeline(config):
    print("\n=== Testing Inference Pipeline ===")
    predictor = QuestionAnsweringPredictor(config)

    # Verify models loaded
    assert predictor.ranker is not None
    assert predictor.reader is not None

    # Run generation
    predictor.generate_submission()

    # Validation
    if os.path.exists(config.SUBMISSION_PATH):
        df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission generated. Rows: {len(df)}")
        assert "example_id" in df.columns
        assert "PredictionString" in df.columns

        # Check format of a few predictions
        if len(df) > 0:
            example_row = df.iloc[0]
            ex_id = example_row["example_id"]
            assert ex_id.endswith("_long") or ex_id.endswith(
                "_short"
            ), "Invalid example_id format"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # Initialize Config
    config = DemoConfig()
    set_seed(config.SEED)

    print(f"Running demo with DEBUG_SAMPLE_SIZE={config.DEBUG_SAMPLE_SIZE}")
    print(f"Working directory: {config.WORKING_DIR}")

    try:
        # 1. Test Vocab Generation
        tokenizer = test_vocab_generation(config)

        # 2. Test Ranker Logic & Training
        test_ranker_model_logic(config, tokenizer)
        test_ranker_training(config)

        # 3. Test Reader Logic & Training
        test_reader_model_logic(config, tokenizer)
        test_reader_training(config)

        # 4. Test Inference
        test_inference_pipeline(config)

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
