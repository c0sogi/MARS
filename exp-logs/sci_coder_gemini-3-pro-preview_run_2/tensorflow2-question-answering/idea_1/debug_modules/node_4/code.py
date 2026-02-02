import os
import torch
import numpy as np
import pandas as pd
import random
import shutil

# Import library components
from library.config import Config
from library.data_utils import (
    Tokenizer,
    extract_candidate_text,
    build_tokenizer,
    build_idf_weights,
)
from library.embeddings import create_embedding_matrix
from library.dataset import NQTrainDataset, NQInferenceDataset, collate_fn
from library.model import BoERanker
from library.short_answer import TFIDFExtractor
from library.trainer import train_model
from library.inference import generate_predictions


def run_demonstration():
    # 1. Setup Configuration for Speed and Reproducibility
    print("--- 1. Setting up Configuration for Fast Demonstration ---")
    Config.setup()

    # Override Config for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample for building vocab/idf
    Config.EMBEDDING_DIM = 16  # Small embedding dimension for speed
    Config.HIDDEN_DIMS = [32]  # Smaller hidden layers
    Config.VOCAB_SIZE = 1000  # Smaller vocab

    # Ensure reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Demonstrating Data Utils
    print("\n--- 2. Demonstrating Data Utils ---")

    # Test extract_candidate_text
    tokens = ["This", "is", "a", "test", "document", "."]
    extracted = extract_candidate_text(tokens, 1, 4)
    print(f"Extracted text: '{extracted}'")
    assert extracted == "is a test", "Extraction logic failed"

    # Test Tokenizer manually
    tokenizer = Tokenizer()
    counts = {"hello": 10, "world": 5, "test": 2}
    tokenizer.fit(counts)
    seq = tokenizer.text_to_sequence(["hello", "unknown", "world"], max_len=5)
    print(f"Token sequence: {seq}")

    # Verify Tokenizer behavior
    assert len(seq) == 5, "Padding logic failed"
    assert seq[0] == tokenizer.word2idx["hello"], "Token mapping failed"
    assert seq[1] == tokenizer.word2idx[Config.UNK_TOKEN], "UNK handling failed"
    assert seq[3] == tokenizer.word2idx[Config.PAD_TOKEN], "Padding token failed"

    # Build Tokenizer from actual data (subset)
    # We use load_cached_data=False to force a build from scratch for demonstration
    print("Building tokenizer from training data subset...")
    real_tokenizer = build_tokenizer(load_cached_data=False, sample_size=50)
    assert (
        real_tokenizer.vocab_size > 2
    ), "Tokenizer should have learned some vocabulary from data"
    print(f"Built tokenizer with vocab size: {real_tokenizer.vocab_size}")

    # Build IDF weights
    print("Computing IDF weights...")
    idf = build_idf_weights(real_tokenizer, load_cached_data=False, sample_size=50)
    assert len(idf) == real_tokenizer.vocab_size, "IDF weights dimension mismatch"
    assert np.all(idf >= 1.0), "IDF weights should be >= 1.0"
    print(f"Computed IDF weights with shape: {idf.shape}")

    # 3. Demonstrating Embeddings
    print("\n--- 3. Demonstrating Embeddings ---")
    # Initialize embeddings (Randomly since glove_path is None)
    emb_matrix = create_embedding_matrix(
        real_tokenizer,
        glove_path=None,
        embedding_dim=Config.EMBEDDING_DIM,
        load_cached_data=False,
    )
    assert emb_matrix.shape == (
        real_tokenizer.vocab_size,
        Config.EMBEDDING_DIM,
    ), "Embedding matrix shape mismatch"

    # Check PAD token embedding is zero vector
    pad_idx = real_tokenizer.word2idx[Config.PAD_TOKEN]
    assert np.all(emb_matrix[pad_idx] == 0), "Padding embedding should be zero vector"
    print("Embedding matrix created and verified.")

    # 4. Demonstrating Datasets
    print("\n--- 4. Demonstrating Datasets ---")

    # Train Dataset
    print("Initializing NQTrainDataset...")
    train_ds = NQTrainDataset(
        metadata_path=Config.TRAIN_META_PATH,
        data_path=Config.TRAIN_DATA_PATH,
        tokenizer=real_tokenizer,
        limit=20,  # Very small limit for speed
        load_cached_data=False,
    )

    if len(train_ds) > 0:
        sample_batch = train_ds[0]
        # sample_batch is a list of dicts (1 positive + N negatives)
        assert isinstance(
            sample_batch, list
        ), "Train dataset should return a list of samples"
        assert len(sample_batch) > 0, "Sample batch should not be empty"
        assert "q_seq" in sample_batch[0], "Missing q_seq in sample"
        assert "label" in sample_batch[0], "Missing label in sample"
        print(f"Train dataset sample count (pos+negs per item): {len(sample_batch)}")

        # Test Collate Function
        batch = collate_fn([train_ds[0], train_ds[1]])
        assert "q_seqs" in batch, "Collate failed to create q_seqs"
        assert "c_seqs" in batch, "Collate failed to create c_seqs"
        assert "labels" in batch, "Collate failed to create labels"
        # Expected batch size = (pos + negs) * 2 items
        expected_size = len(train_ds[0]) + len(train_ds[1])
        assert (
            batch["labels"].shape[0] == expected_size
        ), "Batch size mismatch after flatten"
    else:
        print(
            "Warning: Train dataset empty after filtering (no long answers in the first 20 records)."
        )

    # Inference Dataset
    print("Initializing NQInferenceDataset...")
    inf_ds = NQInferenceDataset(
        metadata_path=Config.TEST_META_PATH,
        data_path=Config.TEST_DATA_PATH,
        tokenizer=real_tokenizer,
        limit=10,
        load_cached_data=False,
    )

    if len(inf_ds) > 0:
        inf_item = inf_ds[0]
        assert "candidates" in inf_item, "Inference item missing candidates"
        assert "example_id" in inf_item, "Inference item missing example_id"
        print(
            f"Inference item ID: {inf_item['example_id']}, Candidate count: {len(inf_item['candidates'])}"
        )

    # 5. Demonstrating Model
    print("\n--- 5. Demonstrating Model ---")
    model = BoERanker(emb_matrix)

    # Create dummy input batch
    bs = 2
    seq_len = Config.MAX_SEQ_LEN
    dummy_q = torch.randint(0, real_tokenizer.vocab_size, (bs, seq_len))
    dummy_c = torch.randint(0, real_tokenizer.vocab_size, (bs, seq_len))

    # Forward pass
    output = model(dummy_q, dummy_c)
    assert output.shape == (bs,), f"Model output shape mismatch: {output.shape}"
    assert torch.all(
        (output >= 0) & (output <= 1)
    ), "Model output not in [0, 1] range (Sigmoid output expected)"
    print("Model forward pass successful.")

    # 6. Demonstrating Short Answer Extractor
    print("\n--- 6. Demonstrating Short Answer Extractor ---")
    # Initialize extractor (will load the cached vocab/idf we built earlier)
    extractor = TFIDFExtractor(load_cached_data=True)

    q_text = "who is the president"
    c_text = "The president is John Doe currently serving his term."

    # Test sliding window search
    # Arbitrary start index 100
    res = extractor.sliding_window_search(q_text, c_text, 100)
    print(f"Short answer extraction result: {res}")
    assert "score" in res and "text" in res, "Invalid extraction result format"
    assert res["start_token"] >= 100, "Start token index incorrect"

    # Test Yes/No logic
    yn_yes = extractor.determine_yes_no("Is the sky blue?", "Yes, it is.")
    assert yn_yes == "YES", "Yes detection failed"

    yn_no = extractor.determine_yes_no("Did he win?", "No, he lost.")
    assert yn_no == "NO", "No detection failed"

    yn_none = extractor.determine_yes_no("Who are you?", "I am a model.")
    assert yn_none == "NONE", "NONE detection failed"
    print("Short Answer Extractor logic verified.")

    # 7. Integrated Training Loop
    print("\n--- 7. Running Training Loop Integration ---")
    # Using a slightly larger limit to ensure we get enough positive samples for a batch
    # The trainer uses Config.TRAIN_DATA_PATH and Config.TRAIN_META_PATH
    trained_model = train_model(load_cached_data=True, limit=50)

    # Verify model checkpoint creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved"
    print("Training loop executed successfully.")

    # 8. Integrated Inference Loop
    print("\n--- 8. Running Inference Loop Integration ---")
    # Generate predictions on a small subset of test data
    generate_predictions(load_cached_data=True, limit=20)

    # Verify submission file creation
    assert os.path.exists(
        Config.SUBMISSION_SAVE_PATH
    ), "Submission file was not created"

    # Verify submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_SAVE_PATH)
    assert "example_id" in sub_df.columns, "Submission missing example_id column"
    assert (
        "PredictionString" in sub_df.columns
    ), "Submission missing PredictionString column"
    print(f"Submission file generated with {len(sub_df)} rows.")
    print(sub_df.head())

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
