import os
import json
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import set_seed, load_glove_embeddings, compute_f1
from library.data_processing import build_vocab, Tokenizer
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import FiLMNetwork
from library.trainer import Trainer
from library.inference import Predictor


def create_mock_data(working_dir):
    """
    Creates small mock JSONL files and metadata CSVs to simulate the NQ dataset
    without reading the massive original files.
    """
    print("Creating mock data for demonstration...")

    # Define paths
    mock_train_file = os.path.join(working_dir, "mock_train.jsonl")
    mock_test_file = os.path.join(working_dir, "mock_test.jsonl")
    mock_train_meta = os.path.join(working_dir, "mock_train_meta.csv")
    mock_val_meta = os.path.join(working_dir, "mock_val_meta.csv")
    mock_test_meta = os.path.join(working_dir, "mock_test_meta.csv")

    # Mock Data Content
    # We create a document with tokens 0-50.
    # Candidate 1: 0-10, Candidate 2: 10-20.
    # Question: "test question"

    entries = []
    for i in range(20):
        entry = {
            "example_id": str(1000 + i),
            "document_text": "This is a mock document text for testing purposes. " * 10,
            "question_text": f"Mock question {i}?",
            "long_answer_candidates": [
                {"start_token": 0, "end_token": 10, "top_level": True},
                {"start_token": 10, "end_token": 20, "top_level": True},
                {"start_token": 20, "end_token": 30, "top_level": True},
            ],
            "annotations": [
                {
                    "long_answer": {
                        "candidate_index": 0 if i % 2 == 0 else -1
                    },  # Alternate positive/negative
                    "short_answers": (
                        [{"start_token": 2, "end_token": 5}] if i % 2 == 0 else []
                    ),
                    "yes_no_answer": "NONE",
                }
            ],
        }
        entries.append(entry)

    # Write JSONL
    with open(mock_train_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # Test data (no annotations)
    test_entries = []
    for i in range(5):
        entry = {
            "example_id": str(2000 + i),
            "document_text": "Test document text for inference. " * 5,
            "question_text": f"Test question {i}?",
            "long_answer_candidates": [
                {"start_token": 0, "end_token": 10, "top_level": True},
                {"start_token": 15, "end_token": 25, "top_level": True},
            ],
        }
        test_entries.append(entry)

    with open(mock_test_file, "w") as f:
        for e in test_entries:
            f.write(json.dumps(e) + "\n")

    # Create Metadata CSVs
    # Train Meta
    train_meta_rows = []
    for i in range(15):  # First 15 for train
        e = entries[i]
        ann = e["annotations"][0]
        la_idx = ann["long_answer"]["candidate_index"]
        has_short = len(ann["short_answers"]) > 0
        train_meta_rows.append(
            {
                "example_id": e["example_id"],
                "long_answer_index": la_idx,
                "has_short_answer": has_short,
                "yes_no_answer": "NONE",
                "stratify_label": "mock",
                "file_path": "mock_train.jsonl",  # Relative to input dir in real code, but we will hack paths
            }
        )
    pd.DataFrame(train_meta_rows).to_csv(mock_train_meta, index=False)

    # Val Meta
    val_meta_rows = []
    for i in range(15, 20):  # Last 5 for val
        e = entries[i]
        ann = e["annotations"][0]
        la_idx = ann["long_answer"]["candidate_index"]
        has_short = len(ann["short_answers"]) > 0
        val_meta_rows.append(
            {
                "example_id": e["example_id"],
                "long_answer_index": la_idx,
                "has_short_answer": has_short,
                "yes_no_answer": "NONE",
                "stratify_label": "mock",
                "file_path": "mock_train.jsonl",
            }
        )
    pd.DataFrame(val_meta_rows).to_csv(mock_val_meta, index=False)

    # Test Meta
    test_meta_rows = []
    for e in test_entries:
        test_meta_rows.append(
            {"example_id": e["example_id"], "file_path": "mock_test.jsonl"}
        )
    pd.DataFrame(test_meta_rows).to_csv(mock_test_meta, index=False)

    return (
        mock_train_file,
        mock_test_file,
        mock_train_meta,
        mock_val_meta,
        mock_test_meta,
    )


def run_demonstration():
    # 1. Setup Configuration
    # We override Config paths to point to our mock data in ./working
    demo_dir = "./working/demo_run"
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Cache files
    Config.VOCAB_FILE = os.path.join(Config.CACHE_DIR, "vocab.npy")
    Config.EMBEDDING_MATRIX_FILE = os.path.join(
        Config.CACHE_DIR, "embedding_matrix.npy"
    )
    Config.TRAIN_CACHE = os.path.join(Config.CACHE_DIR, "train_features.parquet")
    Config.VAL_CACHE = os.path.join(Config.CACHE_DIR, "val_features.parquet")
    Config.TEST_CACHE = os.path.join(Config.CACHE_DIR, "test_features.parquet")

    # Hyperparameters for speed
    Config.VOCAB_SIZE = 500
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.EMBEDDING_DIM = 32
    Config.CNN_FILTERS = 16
    Config.FILM_DIM = 16
    Config.MAX_CTX_LEN = 50
    Config.MAX_Q_LEN = 10

    Config.setup()

    # Create Mock Data
    train_file, test_file, train_meta, val_meta, test_meta = create_mock_data(demo_dir)

    # IMPORTANT: The library code expects file_path in metadata to be relative to Config.INPUT_DIR.
    # Since we put mock files in demo_dir, we must set INPUT_DIR to demo_dir.
    Config.INPUT_DIR = demo_dir
    Config.TRAIN_FILE = train_file
    Config.TEST_FILE = test_file
    Config.TRAIN_META = train_meta
    Config.VAL_META = val_meta
    Config.TEST_META = test_meta

    set_seed(Config.SEED)

    # 2. Vocabulary Building
    print("\n--- Building Vocabulary ---")
    # Force rebuild to use mock data
    vocab = build_vocab(load_cached_data=False)
    assert len(vocab) > 2, "Vocabulary should contain more than just special tokens"
    print(f"Vocabulary size: {len(vocab)}")

    # 3. Embeddings
    print("\n--- Loading Embeddings ---")
    embedding_matrix = load_glove_embeddings(
        vocab.stoi, Config.EMBEDDING_DIM, glove_path=None
    )
    assert embedding_matrix.shape == (len(vocab), Config.EMBEDDING_DIM)

    # 4. DataLoaders
    print("\n--- Preparing DataLoaders ---")
    # Force recompute cache
    if os.path.exists(Config.TRAIN_CACHE):
        os.remove(Config.TRAIN_CACHE)
    if os.path.exists(Config.VAL_CACHE):
        os.remove(Config.VAL_CACHE)

    train_loader, val_loader = get_dataloaders(vocab, load_cached_data=False)

    # Verify Train Batch
    batch = next(iter(train_loader))
    print(f"Train Batch Keys: {batch.keys()}")
    assert "q_input" in batch
    assert "pos_cand_input" in batch
    assert batch["q_input"].shape == (Config.BATCH_SIZE, Config.MAX_Q_LEN)
    assert batch["pos_cand_input"].shape == (Config.BATCH_SIZE, Config.MAX_CTX_LEN)

    # Verify Val Batch (Batch size 1, multiple candidates)
    val_batch = next(iter(val_loader))
    print(f"Val Batch Candidates Shape: {val_batch['candidates'].shape}")
    # Shape should be (1, Num_Cands, Ctx_Len)
    assert len(val_batch["candidates"].shape) == 3

    # 5. Model Initialization
    print("\n--- Initializing Model ---")
    device = torch.device("cpu")  # Use CPU for demo stability/simplicity
    model = FiLMNetwork(embedding_matrix)
    model.to(device)

    # Test Forward Pass
    print("Testing forward pass...")
    with torch.no_grad():
        out = model(batch["q_input"].to(device), batch["pos_cand_input"].to(device))

    assert "rank_logits" in out
    assert "start_logits" in out
    assert out["rank_logits"].shape == (Config.BATCH_SIZE, 1)
    assert out["start_logits"].shape == (Config.BATCH_SIZE, Config.MAX_CTX_LEN)
    print("Forward pass successful.")

    # 6. Training Loop
    print("\n--- Starting Training ---")
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."

    # 7. Inference
    print("\n--- Running Inference ---")
    # Clean test cache
    if os.path.exists(Config.TEST_CACHE):
        os.remove(Config.TEST_CACHE)

    predictor = Predictor(device=device, model_path=Config.MODEL_SAVE_PATH)

    # Manually inject the vocab we built, since Predictor builds its own from config paths
    # which we have set correctly, but we want to ensure consistency in this script flow.
    predictor.vocab = vocab

    results = predictor.generate_predictions(
        threshold=0.0
    )  # Low threshold to force output

    assert len(results) > 0, "No predictions generated"
    print(f"Generated {len(results)} prediction rows.")
    print("Sample prediction:", results[0])

    predictor.save_submission(results)
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created."

    # 8. Metric Utility Check
    print("\n--- Verifying Metrics ---")
    f1 = compute_f1((0, 5), (0, 5))
    assert f1 == 1.0
    f1_partial = compute_f1((0, 5), (0, 2))
    assert 0.0 < f1_partial < 1.0
    print("Metric check passed.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
