import os
import shutil
import torch
import numpy as np
import pandas as pd
import random

# Import from the provided library
from library.config import Config
from library.vocab import Vocabulary
from library.dataset import NQDataset, collate_fn
from library.model import AGBoEModel
from library.trainer import Trainer
from library.inference import InferenceEngine


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_config():
    """
    Modifies the global Config class to optimize for a quick demonstration run.
    Uses a temporary directory and reduces data size/epochs.
    """
    print("--- Setting up Demo Configuration ---")

    # Use a specific demo directory to avoid conflicts with existing caches
    Config.WORKING_DIR = "./working/demo_run_idea_11"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Update cache paths based on new CACHE_DIR
    Config.VOCAB_CACHE_PATH = os.path.join(Config.CACHE_DIR, "vocab.npy")
    Config.EMBEDDING_MATRIX_CACHE_PATH = os.path.join(
        Config.CACHE_DIR, "embedding_matrix.npy"
    )
    Config.TRAIN_FEATURES_CACHE_PATH = os.path.join(
        Config.CACHE_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_CACHE_PATH = os.path.join(
        Config.CACHE_DIR, "val_features.parquet"
    )
    Config.TEST_FEATURES_CACHE_PATH = os.path.join(
        Config.CACHE_DIR, "test_features.parquet"
    )

    # Reduce hyperparameters for speed
    Config.MAX_VOCAB_SIZE = 1000  # Small vocab
    Config.EMBEDDING_DIM = 32  # Small embeddings
    Config.HIDDEN_DIM = 32
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 examples

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")


def demonstrate_vocab():
    print("\n--- Demonstrating Vocabulary ---")

    # Force build from scratch for demo
    vocab = Vocabulary()
    vocab.build_from_corpus(
        Config.TRAIN_DATA_PATH, sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Assertions
    assert len(vocab) > 2, "Vocabulary should contain at least PAD and UNK tokens"
    assert vocab.lookup_word(0) == Config.PAD_TOKEN, "Index 0 should be PAD"

    # Test tokenization
    text = "what is the capital of france"
    indices = vocab.text_to_indices(text, max_length=10)
    assert len(indices) == 10, "Tokenization should pad to max_length"
    assert all(isinstance(i, int) for i in indices), "Indices should be integers"

    # Test Embeddings
    embedding_matrix = vocab.create_embedding_matrix()
    assert embedding_matrix.shape == (
        len(vocab),
        Config.EMBEDDING_DIM,
    ), "Embedding matrix shape mismatch"

    # Save manually to ensure cache exists for subsequent steps
    vocab.save(
        Config.VOCAB_CACHE_PATH, Config.EMBEDDING_MATRIX_CACHE_PATH, embedding_matrix
    )
    print("Vocabulary and Embeddings verified and saved.")
    return vocab, embedding_matrix


def demonstrate_dataset(vocab):
    print("\n--- Demonstrating Dataset ---")

    # Initialize Dataset (Train)
    # We use load_cached_data=False to force processing the small sample
    train_ds = NQDataset(
        split="train",
        vocab=vocab,
        load_cached_data=False,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    print(f"Dataset size: {len(train_ds)}")
    if len(train_ds) == 0:
        print(
            "Warning: Dataset is empty. This might happen if metadata doesn't match raw file sample."
        )
        return

    # Check single item
    item = train_ds[0]
    required_keys = [
        "q_indices",
        "c_indices",
        "label",
        "attn_mask",
        "yes_no",
        "cand_global_start",
        "cand_global_end",
    ]
    for key in required_keys:
        assert key in item, f"Missing key in dataset item: {key}"

    assert item["q_indices"].shape == (Config.MAX_SEQ_LEN_Q,), "Question shape mismatch"
    assert item["c_indices"].shape == (
        Config.MAX_SEQ_LEN_C,
    ), "Candidate shape mismatch"

    # Test Collate Function
    batch_list = [train_ds[i] for i in range(min(len(train_ds), 4))]
    batch = collate_fn(batch_list)

    assert batch["q_indices"].shape == (len(batch_list), Config.MAX_SEQ_LEN_Q)
    assert batch["labels"].shape == (len(batch_list),)
    assert len(batch["example_ids"]) == len(batch_list)

    print("Dataset and Collate function verified.")


def demonstrate_model(embedding_matrix):
    print("\n--- Demonstrating Model ---")

    model = AGBoEModel(embedding_matrix)

    # Create dummy batch
    batch_size = 2
    q_indices = torch.randint(
        0, len(embedding_matrix), (batch_size, Config.MAX_SEQ_LEN_Q)
    )
    c_indices = torch.randint(
        0, len(embedding_matrix), (batch_size, Config.MAX_SEQ_LEN_C)
    )

    # Forward pass
    ranking_logits, yesno_logits, attn_weights = model(q_indices, c_indices)

    # Verification
    assert ranking_logits.shape == (
        batch_size,
    ), f"Ranking logits shape mismatch: {ranking_logits.shape}"
    assert yesno_logits.shape == (
        batch_size,
        3,
    ), f"YesNo logits shape mismatch: {yesno_logits.shape}"
    assert attn_weights.shape == (
        batch_size,
        Config.MAX_SEQ_LEN_C,
    ), f"Attn weights shape mismatch: {attn_weights.shape}"

    print("Model architecture and forward pass verified.")


def demonstrate_training():
    print("\n--- Demonstrating Training Loop ---")

    # Initialize Trainer
    # We pass sample_size to limit data loading
    trainer = Trainer(load_cached_data=True, sample_size=Config.DEBUG_SAMPLE_SIZE)

    # Run training (1 epoch as configured in setup_demo_config)
    trainer.train()

    # Check if model was saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training loop completed and model saved.")


def demonstrate_inference():
    print("\n--- Demonstrating Inference Pipeline ---")

    # Initialize Inference Engine
    engine = InferenceEngine(
        load_cached_data=False, sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Run inference
    engine.run_inference()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission rows: {len(df)}")
    assert "example_id" in df.columns
    assert "PredictionString" in df.columns

    # Basic format check
    if len(df) > 0:
        ex_id = df.iloc[0]["example_id"]
        assert ex_id.endswith("_long") or ex_id.endswith(
            "_short"
        ), "Invalid example_id format"

    print("Inference pipeline completed successfully.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    setup_demo_config()

    # 2. Vocab
    vocab, embedding_matrix = demonstrate_vocab()

    # 3. Dataset
    demonstrate_dataset(vocab)

    # 4. Model
    demonstrate_model(embedding_matrix)

    # 5. Training
    # Note: Trainer will reload vocab/embeddings from the cache we just created
    demonstrate_training()

    # 6. Inference
    demonstrate_inference()

    print("\nAll demonstrations completed successfully.")
