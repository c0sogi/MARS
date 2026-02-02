import os
import json
import shutil
import numpy as np
import pandas as pd
import torch
import random
from library.config import Config
from library.vocab_manager import VocabManager
from library.window_processor import WindowProcessor
from library.data_loader import get_data_loaders
from library.model import WindowMaxPoolingNetwork
from library.solver import Solver


# --- 1. Setup Demo Configuration ---
class DemoConfig(Config):
    """
    Configuration override for a fast demonstration run.
    """

    WORKING_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Mock Data Paths
    TRAIN_DATA_PATH = os.path.join(WORKING_DIR, "mock_train.jsonl")
    TEST_DATA_PATH = os.path.join(WORKING_DIR, "mock_test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Mock Metadata Paths
    TRAIN_META_PATH = os.path.join(WORKING_DIR, "mock_train_meta.csv")
    VAL_META_PATH = os.path.join(WORKING_DIR, "mock_val_meta.csv")
    TEST_META_PATH = os.path.join(WORKING_DIR, "mock_test_meta.csv")

    # Cache Paths
    VOCAB_PATH = os.path.join(CACHE_DIR, "vocab.npy")
    EMBEDDING_MATRIX_PATH = os.path.join(CACHE_DIR, "embedding_matrix.npy")

    # Model Paths
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    FINAL_SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Reduced Parameters for Speed
    DEBUG = True
    DEBUG_SIZE = 100  # Process small number of samples
    VOCAB_SIZE = 500
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 64
    WINDOW_SIZE = 32
    WINDOW_STRIDE = 16
    MAX_QUESTION_LEN = 10
    BATCH_SIZE = 4
    NUM_EPOCHS = 1
    LEARNING_RATE = 1e-3
    NEGATIVE_SAMPLING_RATIO = 1.0  # Reduce negatives for speed


def setup_demo_environment(config):
    """Creates directories and generates mock data."""
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR)
    os.makedirs(config.CACHE_DIR)
    os.makedirs(config.SUBMISSION_DIR)

    # --- Generate Mock Train Data ---
    train_data = []
    # Create 20 training examples
    for i in range(20):
        doc_text = (
            "this is a sample document text with some random words repeated " * 20
        )
        # Ensure the answer is somewhere in the text
        doc_text += " the answer is here " + " filler " * 10

        # Candidate spanning the answer
        start_token = 120
        end_token = 124  # "the answer is here"

        entry = {
            "example_id": str(1000 + i),
            "document_text": doc_text,
            "question_text": "where is the answer located",
            "long_answer_candidates": [
                {"start_token": 0, "end_token": 50, "top_level": True},
                {
                    "start_token": 100,
                    "end_token": 150,
                    "top_level": True,
                },  # Contains answer
                {"start_token": 200, "end_token": 250, "top_level": True},
            ],
            "annotations": [
                {
                    "long_answer": {
                        "candidate_index": 1
                    },  # The second candidate is correct
                    "short_answers": [{"start_token": 120, "end_token": 124}],
                    "yes_no_answer": "NONE",
                }
            ],
        }
        train_data.append(entry)

    with open(config.TRAIN_DATA_PATH, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    # --- Generate Mock Test Data ---
    test_data = []
    # Create 5 test examples
    for i in range(5):
        entry = {
            "example_id": str(2000 + i),
            "document_text": "test document text for inference " * 30,
            "question_text": "what is the test prediction",
            "long_answer_candidates": [
                {"start_token": 10, "end_token": 60, "top_level": True},
                {"start_token": 70, "end_token": 120, "top_level": True},
            ],
        }
        test_data.append(entry)

    with open(config.TEST_DATA_PATH, "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")

    # --- Generate Mock Sample Submission ---
    sample_sub_rows = []
    for i in range(5):
        eid = str(2000 + i)
        sample_sub_rows.append({"example_id": f"{eid}_long", "PredictionString": ""})
        sample_sub_rows.append({"example_id": f"{eid}_short", "PredictionString": ""})
    pd.DataFrame(sample_sub_rows).to_csv(config.SAMPLE_SUBMISSION_PATH, index=False)

    # --- Generate Metadata Files ---
    # Split 20 train samples: 15 train, 5 val
    train_ids = [str(1000 + i) for i in range(15)]
    val_ids = [str(1000 + i) for i in range(15, 20)]
    test_ids = [str(2000 + i) for i in range(5)]

    # Train Metadata
    train_meta = pd.DataFrame(
        {
            "example_id": train_ids,
            "file_path": "mock_train.jsonl",  # Relative path logic in loader usually assumes ./input, but we override loader logic via config paths mostly.
            # However, the loader splits based on ID presence, so file_path content here is less critical for the split logic.
            "long_answer_index": [1] * 15,
            "has_short_answer": [True] * 15,
            "yes_no_answer": ["NONE"] * 15,
            "stratify_label": ["L1_S1_NONE"] * 15,
        }
    )
    train_meta.to_csv(config.TRAIN_META_PATH, index=False)

    # Val Metadata
    val_meta = pd.DataFrame(
        {
            "example_id": val_ids,
            "file_path": "mock_train.jsonl",
            "long_answer_index": [1] * 5,
            "has_short_answer": [True] * 5,
            "yes_no_answer": ["NONE"] * 5,
            "stratify_label": ["L1_S1_NONE"] * 5,
        }
    )
    val_meta.to_csv(config.VAL_META_PATH, index=False)

    # Test Metadata
    test_meta = pd.DataFrame({"example_id": test_ids, "file_path": "mock_test.jsonl"})
    test_meta.to_csv(config.TEST_META_PATH, index=False)

    print("Demo environment setup complete.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # 1. Setup
    config = DemoConfig()
    set_seed(config.SEED)
    setup_demo_environment(config)

    print("\n--- 1. Testing VocabManager ---")
    vocab_manager = VocabManager(config)
    # Build vocab from scratch (force recompute)
    vocab_manager.build_vocab(load_cached_data=False)

    vocab_size = vocab_manager.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    assert (
        vocab_size > 2
    ), "Vocabulary should contain more than just PAD and UNK tokens."

    emb_matrix = vocab_manager.get_embedding_matrix()
    assert emb_matrix.shape == (
        vocab_size,
        config.EMBEDDING_DIM,
    ), "Embedding matrix shape mismatch."

    # Test text to indices
    indices = vocab_manager.text_to_indices("this is a sample")
    print(f"Indices for 'this is a sample': {indices}")
    assert len(indices) == 4, "Tokenization length mismatch."

    print("\n--- 2. Testing WindowProcessor ---")
    processor = WindowProcessor(config, vocab_manager)

    # Load one example to test processing logic manually
    with open(config.TRAIN_DATA_PATH, "r") as f:
        first_line = json.loads(f.readline())

    features = processor.process_single_example(first_line, is_train=True)
    print(f"Generated {len(features)} windows for the first example.")

    if len(features) > 0:
        feat = features[0]
        print(f"Feature keys: {feat.keys()}")
        assert "input_ids" in feat
        assert "label_window" in feat
        assert len(feat["input_ids"]) == config.WINDOW_SIZE, "Window size mismatch."

        # Check if we have positive labels (since we engineered the data to have them)
        has_positive = any(f["label_window"] == 1 for f in features)
        print(f"Contains positive window: {has_positive}")
        assert has_positive, "Expected at least one positive window in the mock data."

    print("\n--- 3. Testing Data Loaders ---")
    # This will process the dataset and create cache files
    train_loader, val_loader, test_loader = get_data_loaders(
        config, vocab_manager, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")

    assert batch["input_ids"].shape[1] == config.WINDOW_SIZE
    assert batch["question_ids"].shape[1] == config.MAX_QUESTION_LEN
    assert batch["label_window"].dtype == torch.float32

    print("\n--- 4. Testing Model Architecture ---")
    model = WindowMaxPoolingNetwork(emb_matrix, config)

    # Forward pass with the batch
    window_score, start_logits, end_logits, yes_no_logits = model(
        batch["input_ids"], batch["question_ids"]
    )

    print(f"Window Score shape: {window_score.shape}")
    print(f"Start Logits shape: {start_logits.shape}")
    print(f"Yes/No Logits shape: {yes_no_logits.shape}")

    assert window_score.shape == (batch["input_ids"].size(0), 1)
    assert start_logits.shape == (batch["input_ids"].size(0), config.WINDOW_SIZE)
    assert yes_no_logits.shape == (
        batch["input_ids"].size(0),
        config.NUM_YES_NO_CLASSES,
    )

    print("\n--- 5. Testing Solver (Training Loop) ---")
    solver = Solver(model, config)

    # Train for 1 epoch
    solver.train(train_loader, val_loader)

    # Check if model checkpoint exists
    assert os.path.exists(
        config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved."
    print("Model training and checkpointing successful.")

    print("\n--- 6. Testing Inference ---")
    # Run inference on test set
    solver.inference(test_loader)

    # Check if submission file exists
    assert os.path.exists(
        config.FINAL_SUBMISSION_PATH
    ), "Submission file was not generated."

    # Verify submission content
    sub_df = pd.read_csv(config.FINAL_SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    # We expect 5 examples * 2 rows (long/short) = 10 rows
    assert len(sub_df) == 10, f"Expected 10 rows in submission, got {len(sub_df)}"

    print("\nAll demonstrations completed successfully.")
