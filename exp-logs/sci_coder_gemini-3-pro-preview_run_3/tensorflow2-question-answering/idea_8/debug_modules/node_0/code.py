import os
import torch
import pandas as pd
import numpy as np
import random
import shutil

# Import from the provided library
from library.config import Config
from library.utils import (
    tokenize_text,
    normalize_answer,
    exact_match_score,
    f1_score,
    parse_html_candidates,
    load_glove_embeddings,
)
from library.data import (
    Vocabulary,
    get_vocabulary,
    get_ranker_data,
    get_reader_data,
    get_test_candidates,
    NQRankerDataset,
    NQReaderDataset,
)
from library.models import ANBoWRanker, ConvBiDAFReader
from library.trainer import Trainer
from library.inference import InferencePipeline


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_config():
    """
    Modifies the global Config class to run a fast demonstration.
    """
    print("Setting up demonstration configuration...")

    # Use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update paths based on new working dir
    Config.VOCAB_CACHE_PATH = os.path.join(Config.WORKING_DIR, "vocab.parquet")
    Config.EMBEDDING_MATRIX_PATH = os.path.join(
        Config.WORKING_DIR, "embedding_matrix.npy"
    )
    Config.RANKER_TRAIN_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "ranker_train_data.parquet"
    )
    Config.RANKER_VAL_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "ranker_val_data.parquet"
    )
    Config.READER_TRAIN_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "reader_train_data.parquet"
    )
    Config.READER_VAL_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "reader_val_data.parquet"
    )
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "ranker_test_features.parquet"
    )
    Config.RANKER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ranker_best.pth")
    Config.READER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "reader_best.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce data size for speed
    Config.TRAIN_SAMPLE_SIZE = 200
    Config.VAL_SAMPLE_SIZE = 50

    # Reduce model complexity for speed
    Config.VOCAB_SIZE = 2000
    Config.EMBEDDING_DIM = 32
    Config.RANKER_HIDDEN_DIM = 32
    Config.READER_HIDDEN_DIM = 32

    # Training settings
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Re-create directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup_directories()


def demo_utils():
    print("\n--- Demonstrating Library Utils ---")

    # 1. Tokenization
    text = "Hello, world! This is a test."
    tokens = tokenize_text(text)
    print(f"Tokens: {tokens}")
    assert len(tokens) == 6, "Tokenization failed length check"

    # 2. Normalization
    raw_ans = "The United States of America."
    norm_ans = normalize_answer(raw_ans)
    print(f"Normalized: '{norm_ans}'")
    assert norm_ans == "united states of america", "Normalization logic incorrect"

    # 3. Metrics
    em = exact_match_score("United States", "The United States")
    f1 = f1_score("United States", "United Kingdom")
    print(f"EM Score: {em}, F1 Score: {f1:.4f}")
    assert em == 1.0, "EM score should be 1.0 after normalization"
    assert f1 > 0.0, "F1 score should be positive for overlap"

    # 4. HTML Parsing
    html_tokens = [
        "<P>",
        "This",
        "is",
        "para",
        "1",
        "</P>",
        "<Table>",
        "<Tr>",
        "Data",
        "</Tr>",
        "</Table>",
    ]
    candidates = parse_html_candidates(html_tokens)
    print(f"Candidates found: {len(candidates)}")
    assert len(candidates) == 2, "Should detect 2 top-level candidates"
    assert candidates[0] == (0, 6), "First candidate span incorrect"


def demo_data_processing():
    print("\n--- Demonstrating Data Processing ---")

    # 1. Vocabulary
    vocab = get_vocabulary(load_cached_data=False)
    print(f"Vocabulary built. Size: {len(vocab.token_to_idx)}")
    assert len(vocab.token_to_idx) > 2, "Vocab should contain more than special tokens"

    # 2. Ranker Data Generation
    print("Generating Ranker Data...")
    ranker_train_df = get_ranker_data(split="train", load_cached_data=False)
    print(f"Ranker Train Data Shape: {ranker_train_df.shape}")
    if not ranker_train_df.empty:
        assert "label" in ranker_train_df.columns
        dataset = NQRankerDataset(ranker_train_df, vocab)
        sample = dataset[0]
        assert "q_ids" in sample and "c_ids" in sample
        assert sample["q_ids"].shape[0] == Config.MAX_Q_LEN

    # 3. Reader Data Generation
    print("Generating Reader Data...")
    reader_train_df = get_reader_data(split="train", load_cached_data=False)
    print(f"Reader Train Data Shape: {reader_train_df.shape}")
    if not reader_train_df.empty:
        assert "start_idx" in reader_train_df.columns
        dataset = NQReaderDataset(reader_train_df, vocab)
        sample = dataset[0]
        assert "start_idx" in sample and "end_idx" in sample


def demo_models():
    print("\n--- Demonstrating Models ---")

    # Setup dummy inputs
    batch_size = 4
    q_len = Config.MAX_Q_LEN
    c_len = Config.MAX_DOC_LEN
    vocab_size = Config.VOCAB_SIZE

    q_ids = torch.randint(0, vocab_size, (batch_size, q_len))
    c_ids = torch.randint(0, vocab_size, (batch_size, c_len))

    # Generate random embedding matrix for initialization
    embedding_matrix = np.random.randn(vocab_size, Config.EMBEDDING_DIM).astype(
        np.float32
    )

    # 1. Ranker
    print("Initializing ANBoWRanker...")
    ranker = ANBoWRanker(embedding_matrix=embedding_matrix)
    ranker.eval()
    with torch.no_grad():
        logits = ranker(q_ids, c_ids)
    print(f"Ranker Output Shape: {logits.shape}")
    assert logits.shape == (batch_size,), "Ranker output shape mismatch"

    # 2. Reader
    print("Initializing ConvBiDAFReader...")
    reader = ConvBiDAFReader(embedding_matrix=embedding_matrix)
    reader.eval()
    with torch.no_grad():
        start_logits, end_logits = reader(q_ids, c_ids)
    print(f"Reader Start Logits Shape: {start_logits.shape}")
    assert start_logits.shape == (
        batch_size,
        c_len,
    ), "Reader start logits shape mismatch"
    assert end_logits.shape == (batch_size, c_len), "Reader end logits shape mismatch"


def demo_training():
    print("\n--- Demonstrating Training Loop ---")

    trainer = Trainer()

    # Train Ranker (1 epoch, small data)
    print("Training Ranker...")
    trainer.train_ranker()
    assert os.path.exists(Config.RANKER_MODEL_PATH), "Ranker model checkpoint not saved"

    # Train Reader (1 epoch, small data)
    print("Training Reader...")
    trainer.train_reader()
    assert os.path.exists(Config.READER_MODEL_PATH), "Reader model checkpoint not saved"


def demo_inference():
    print("\n--- Demonstrating Inference Pipeline ---")

    pipeline = InferencePipeline()

    # Run inference on test set
    # Using load_cached_data=False to force processing of test candidates
    pipeline.run_inference(load_cached_data=False, batch_size=Config.BATCH_SIZE)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify submission format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission rows: {len(df)}")
    assert "example_id" in df.columns and "PredictionString" in df.columns
    assert len(df) > 0, "Submission dataframe is empty"


if __name__ == "__main__":
    set_seed(42)

    # 1. Setup environment for demo
    setup_demo_config()

    # 2. Run demonstrations
    try:
        demo_utils()
        demo_data_processing()
        demo_models()
        demo_training()
        demo_inference()
        print("\nAll demonstrations completed successfully!")
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        raise e
