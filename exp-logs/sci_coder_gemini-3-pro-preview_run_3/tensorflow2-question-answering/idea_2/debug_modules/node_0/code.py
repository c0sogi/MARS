import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.text_processing import Tokenizer, HTMLParser, build_vocab
from library.datasets import (
    prepare_ranker_data,
    prepare_reader_data,
    RankerDataset,
    ReaderDataset,
)
from library.networks import SiameseTextCNN, AttentionMLPReader
from library.training_engine import train_ranker, train_reader
from library.inference_pipeline import InferencePipeline, generate_submission


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Use a separate working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.setup_directories()

    # Override paths to point to the new working directory
    Config.VOCAB_CACHE_PATH = os.path.join(Config.WORKING_DIR, "vocab.parquet")
    Config.RANKER_TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "ranker_train_data.parquet"
    )
    Config.RANKER_VAL_CACHE = os.path.join(
        Config.WORKING_DIR, "ranker_val_data.parquet"
    )
    Config.READER_TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "reader_train_data.parquet"
    )
    Config.READER_VAL_CACHE = os.path.join(
        Config.WORKING_DIR, "reader_val_data.parquet"
    )
    Config.RANKER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ranker_best.pth")
    Config.READER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "reader_best.pth")

    # Reduce hyperparameters for speed
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.VOCAB_SIZE = 1000  # Smaller vocabulary
    Config.EMBED_DIM = 32  # Smaller embeddings
    Config.RANKER_HIDDEN_DIM = 32
    Config.READER_HIDDEN_DIM = 32
    Config.CNN_FILTERS = 16

    set_seed(Config.SEED)
    print("Configuration updated.")

    # --------------------------------------------------------------------------
    # 2. Text Processing Demonstration
    # --------------------------------------------------------------------------
    print("\n[2] Testing Text Processing Components...")

    # Test HTMLParser
    parser = HTMLParser()
    doc_text = "This is a <P> sample paragraph </P> . Another <H1> header </H1> ."
    candidates_data = [
        {"start_token": 0, "end_token": 7, "top_level": False},
        {"start_token": 8, "end_token": 12, "top_level": True},
    ]
    candidates = parser.extract_candidates(doc_text, candidates_data)

    assert len(candidates) == 2, "Parser should extract 2 candidates"
    assert (
        candidates[0]["text"] == "This is a <P> sample paragraph </P>"
    ), "First candidate text mismatch"
    assert candidates[1]["top_level"] is True, "Second candidate should be top level"
    print("HTMLParser logic verified.")

    # Test Tokenizer & Vocab Building
    # We force build_vocab to run on the small sample defined in Config
    tokenizer = build_vocab(load_cached_data=False)

    test_sentence = "the quick brown fox"
    seq = tokenizer.text_to_sequence(test_sentence)
    padded_seq = tokenizer.pad_sequence(seq, max_len=10)

    assert len(padded_seq) == 10, "Padding length incorrect"
    assert tokenizer.vocab_size > 0, "Vocabulary size should be positive"
    print(f"Tokenizer verified. Vocab size: {tokenizer.vocab_size}")

    # --------------------------------------------------------------------------
    # 3. Dataset Preparation Demonstration
    # --------------------------------------------------------------------------
    print("\n[3] Testing Dataset Preparation...")

    # Prepare Ranker Data
    # Note: We use load_cached_data=False to force processing logic execution
    ranker_train_ds = prepare_ranker_data(split="train", load_cached_data=False)
    assert isinstance(ranker_train_ds, RankerDataset), "Should return RankerDataset"
    if len(ranker_train_ds) > 0:
        sample_item = ranker_train_ds[0]
        assert (
            "question" in sample_item
            and "paragraph" in sample_item
            and "label" in sample_item
        )
        assert sample_item["question"].shape[0] == Config.MAX_Q_LEN
        assert sample_item["paragraph"].shape[0] == Config.MAX_DOC_LEN
    print(f"Ranker Dataset prepared. Size: {len(ranker_train_ds)}")

    # Prepare Reader Data
    reader_train_ds = prepare_reader_data(split="train", load_cached_data=False)
    assert isinstance(reader_train_ds, ReaderDataset), "Should return ReaderDataset"
    if len(reader_train_ds) > 0:
        sample_item = reader_train_ds[0]
        assert "start_idx" in sample_item and "end_idx" in sample_item
    print(f"Reader Dataset prepared. Size: {len(reader_train_ds)}")

    # --------------------------------------------------------------------------
    # 4. Network Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[4] Testing Network Architectures...")

    device = torch.device("cpu")  # Keep on CPU for shape verification

    # Test SiameseTextCNN (Ranker)
    ranker_model = SiameseTextCNN().to(device)
    dummy_q = torch.randint(0, Config.VOCAB_SIZE, (2, Config.MAX_Q_LEN)).to(device)
    dummy_p = torch.randint(0, Config.VOCAB_SIZE, (2, Config.MAX_DOC_LEN)).to(device)

    with torch.no_grad():
        ranker_out = ranker_model(dummy_q, dummy_p)

    assert ranker_out.shape == (2,), f"Ranker output shape mismatch: {ranker_out.shape}"
    print("SiameseTextCNN forward pass successful.")

    # Test AttentionMLPReader (Reader)
    reader_model = AttentionMLPReader().to(device)

    with torch.no_grad():
        start_logits, end_logits = reader_model(dummy_q, dummy_p)

    assert start_logits.shape == (
        2,
        Config.MAX_DOC_LEN,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        2,
        Config.MAX_DOC_LEN,
    ), f"End logits shape mismatch: {end_logits.shape}"
    print("AttentionMLPReader forward pass successful.")

    # --------------------------------------------------------------------------
    # 5. Training Engine Demonstration
    # --------------------------------------------------------------------------
    print("\n[5] Testing Training Engine...")

    # Since we already prepared data (and cached it via prepare_* functions),
    # train functions will load from cache.

    print("Training Ranker...")
    trained_ranker = train_ranker(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )
    assert os.path.exists(Config.RANKER_MODEL_PATH), "Ranker model checkpoint not found"

    print("Training Reader...")
    trained_reader = train_reader(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )
    assert os.path.exists(Config.READER_MODEL_PATH), "Reader model checkpoint not found"

    print("Training loops completed successfully.")

    # --------------------------------------------------------------------------
    # 6. Inference Pipeline Demonstration
    # --------------------------------------------------------------------------
    print("\n[6] Testing Inference Pipeline...")

    # Initialize pipeline (loads vocab and models we just trained)
    pipeline = InferencePipeline()

    # Run inference on a small subset of the test set
    # Using sample_size=5 to keep it very fast
    generate_submission(sample_size=5)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "example_id" in sub_df.columns
    assert "PredictionString" in sub_df.columns
    assert (
        len(sub_df) == 10
    ), f"Expected 10 rows (5 samples * 2 rows each), got {len(sub_df)}"

    print("Inference pipeline completed successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
