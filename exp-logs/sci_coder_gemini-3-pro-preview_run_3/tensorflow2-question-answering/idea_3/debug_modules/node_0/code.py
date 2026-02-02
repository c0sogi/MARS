import os
import torch
import pandas as pd
import numpy as np
import random
import shutil
from library.config import Config
from library.data_processing import (
    DataProcessor,
    TextProcessor,
    Vocabulary,
    RankerDataset,
    ReaderDataset,
)
from library.models import SiameseRanker, ConditionalReader
from library.trainer import Trainer
from library.inference import InferencePipeline


def setup_environment():
    """
    Overrides Config parameters to ensure the script runs quickly and deterministically
    for demonstration purposes.
    """
    print("Setting up environment and overriding configuration for speed...")

    # Override Config for rapid execution
    Config.TRAIN_SAMPLE_SIZE = 100  # Process only 100 training samples
    Config.VAL_SAMPLE_SIZE = 20  # Process only 20 validation samples
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.VOCAB_SIZE = 500  # Small vocabulary size
    Config.EMBEDDING_DIM = 32  # Reduced embedding dimension
    Config.HIDDEN_DIM = 32  # Reduced hidden dimension
    Config.PATIENCE = 1  # Minimal patience for early stopping

    # Ensure reproducibility
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)

    # Clean up previous run artifacts in working dir to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        for f in os.listdir(Config.WORKING_DIR):
            if f.endswith(".parquet") or f.endswith(".pth"):
                try:
                    os.remove(os.path.join(Config.WORKING_DIR, f))
                except Exception as e:
                    print(f"Note: Could not remove {f}: {e}")


def test_data_processing():
    """
    Verifies DataProcessor, TextProcessor, and Vocabulary functionality.
    """
    print("\n--- Testing Data Processing ---")

    processor = DataProcessor()

    # 1. Test TextProcessor logic
    tp = TextProcessor()
    raw_tokens = ["<H1>", "Title", "Text", "</H1>", "Content"]
    clean_tokens, idx_map = tp.clean_and_map_indices(raw_tokens)

    print(f"Raw tokens: {raw_tokens}")
    print(f"Clean tokens: {clean_tokens}")
    print(f"Index Map: {idx_map}")

    assert "<H1>" not in clean_tokens
    assert "</H1>" not in clean_tokens
    assert "Title" in clean_tokens
    assert len(idx_map) == len(raw_tokens)

    # 2. Verify Data Loading (forcing fresh processing)
    # This will generate cache files in Config.WORKING_DIR
    print("Generating datasets...")
    r_train, r_val, read_train, read_val = processor.get_data(load_cached_data=False)

    print(f"Ranker Train Samples: {len(r_train)}")
    print(f"Reader Train Samples: {len(read_train)}")

    assert len(r_train) > 0, "Ranker training data should not be empty"
    # Note: Reader data might be smaller if samples don't have short answers,
    # but with 100 samples we expect at least some.

    # 3. Verify Vocab
    vocab = processor.vocab
    assert vocab.built, "Vocabulary should be marked as built"
    assert len(vocab) > 2, "Vocabulary should contain more than just special tokens"
    print(f"Vocabulary Size: {len(vocab)}")

    return r_train, vocab


def test_models(sample_data, vocab):
    """
    Verifies Model architectures by running a forward pass.
    """
    print("\n--- Testing Models ---")

    device = torch.device("cpu")  # Use CPU for simple logic check

    # Instantiate Dataset to get a correctly formatted batch item
    ds = RankerDataset(sample_data[:4], vocab)
    batch = ds[0]

    # Prepare batch inputs (unsqueeze to add batch dim)
    q_input = batch["q_input"].unsqueeze(0).to(device)
    ctx_input = batch["ctx_input"].unsqueeze(0).to(device)

    # 1. Test SiameseRanker
    print("Testing SiameseRanker...")
    ranker = SiameseRanker(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
    ).to(device)

    ranker.eval()
    with torch.no_grad():
        scores = ranker(q_input, ctx_input)

    print(f"Ranker Output Shape: {scores.shape}")
    print(f"Ranker Score: {scores.item()}")

    assert scores.shape == (1,), "Ranker output shape mismatch"
    assert -1.0 <= scores.item() <= 1.0, "Cosine similarity out of bounds"

    # 2. Test ConditionalReader
    print("Testing ConditionalReader...")
    reader = ConditionalReader(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
    ).to(device)

    reader.eval()
    with torch.no_grad():
        start_logits, end_logits = reader(q_input, ctx_input)

    print(f"Reader Start Logits Shape: {start_logits.shape}")
    print(f"Reader End Logits Shape: {end_logits.shape}")

    assert start_logits.shape == (
        1,
        Config.MAX_CTX_LEN,
    ), "Reader start logits shape mismatch"
    assert end_logits.shape == (
        1,
        Config.MAX_CTX_LEN,
    ), "Reader end logits shape mismatch"


def test_training():
    """
    Verifies the training loop and checkpoint saving.
    """
    print("\n--- Testing Training Loop ---")

    # Initialize Trainer (loads data using get_data_loaders)
    # We set load_cached_data=True because we generated the cache in test_data_processing
    trainer = Trainer(load_cached_data=True)

    # Run training
    trainer.train_ranker()
    trainer.train_reader()

    # Verify Checkpoints
    assert os.path.exists(Config.RANKER_MODEL_PATH), "Ranker model checkpoint missing"
    assert os.path.exists(Config.READER_MODEL_PATH), "Reader model checkpoint missing"
    print("Training completed and checkpoints verified.")


def test_inference():
    """
    Verifies the inference pipeline and submission generation.
    """
    print("\n--- Testing Inference Pipeline ---")

    # Initialize Pipeline
    pipeline = InferencePipeline()

    # Run generation on a tiny subset of test data
    # We use sample_size=5 to keep it fast
    pipeline.generate_predictions(sample_size=5)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_OUTPUT), "Submission file missing"

    df = pd.read_csv(Config.SUBMISSION_OUTPUT)
    print(f"Submission Rows: {len(df)}")
    print("Submission Head:")
    print(df.head())

    # Basic format check
    assert "example_id" in df.columns
    assert "PredictionString" in df.columns
    # generate_predictions processes sample_size examples.
    # Each example produces 2 rows (long and short).
    # So we expect sample_size * 2 rows.
    expected_rows = 5 * 2
    assert (
        len(df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df)}"


if __name__ == "__main__":
    try:
        setup_environment()

        # Run components sequentially
        ranker_data, vocab = test_data_processing()
        test_models(ranker_data, vocab)
        test_training()
        test_inference()

        print("\nAll demonstrations and verifications passed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        raise e
