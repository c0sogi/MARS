import os
import json
import pandas as pd
import torch
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_vocab, get_dataloaders, get_test_dataloader
from library.model import CQCRNN
import library.engine
import importlib

importlib.reload(library.engine)
from library.engine import train_model
from library.inference import run_inference_pipeline


def create_mock_data(source_path, dest_path, num_lines=50):
    """Creates a smaller version of the JSONL data file for demonstration."""
    print(f"Creating mock data: {dest_path} from {source_path} ({num_lines} lines)...")
    data = []
    with open(source_path, "r", encoding="utf-8") as f_in, open(
        dest_path, "w", encoding="utf-8"
    ) as f_out:
        for i, line in enumerate(f_in):
            if i >= num_lines:
                break
            f_out.write(line)
            data.append(json.loads(line))
    return data


def generate_mock_metadata(json_data, meta_path, is_train=True):
    """Generates metadata CSV corresponding to the mock JSONL data."""
    print(f"Generating mock metadata: {meta_path}...")
    rows = []
    for entry in json_data:
        row = {
            "example_id": str(entry["example_id"]),
            "file_path": (
                os.path.basename(Config.TRAIN_DATA_PATH)
                if is_train
                else os.path.basename(Config.TEST_DATA_PATH)
            ),
        }

        if is_train:
            # Extract labels for training metadata
            anns = entry.get("annotations", [])
            long_idx = -1
            has_short = False
            yes_no = "NONE"

            if anns:
                ann = anns[0]
                long_idx = ann.get("long_answer", {}).get("candidate_index", -1)
                has_short = bool(ann.get("short_answers"))
                yes_no = ann.get("yes_no_answer", "NONE")

            row["long_answer_index"] = long_idx
            row["has_short_answer"] = has_short
            row["yes_no_answer"] = yes_no
            # Create a dummy stratify label
            row["stratify_label"] = "mock_label"

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(meta_path, index=False)
    return df


def setup_demo_environment():
    """Sets up a temporary working directory and overrides Config."""
    # 1. Define paths
    base_dir = "./working/demo_run"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    submission_dir = os.path.join(base_dir, "submission")
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Create Mock Data
    # We use the real input files to sample valid JSON structures
    mock_train_path = os.path.join(base_dir, "mock_train.jsonl")
    mock_test_path = os.path.join(base_dir, "mock_test.jsonl")

    train_data = create_mock_data(Config.TRAIN_DATA_PATH, mock_train_path, num_lines=20)
    test_data = create_mock_data(Config.TEST_DATA_PATH, mock_test_path, num_lines=10)

    # 3. Create Mock Metadata
    mock_train_meta_path = os.path.join(base_dir, "mock_train_meta.csv")
    mock_val_meta_path = os.path.join(base_dir, "mock_val_meta.csv")
    mock_test_meta_path = os.path.join(base_dir, "mock_test_meta.csv")

    # Split train data for train/val metadata
    # First 15 for train, last 5 for val
    train_df = generate_mock_metadata(
        train_data[:15], mock_train_meta_path, is_train=True
    )
    val_df = generate_mock_metadata(train_data[15:], mock_val_meta_path, is_train=True)
    test_df = generate_mock_metadata(test_data, mock_test_meta_path, is_train=False)

    # 4. Override Config
    print("Overriding Config parameters for demo...")
    Config.WORKING_DIR = base_dir
    Config.CACHE_DIR = cache_dir
    Config.SUBMISSION_DIR = submission_dir

    # Point to mock data
    Config.TRAIN_DATA_PATH = mock_train_path
    Config.TEST_DATA_PATH = mock_test_path

    # Point to mock metadata
    Config.TRAIN_META_PATH = mock_train_meta_path
    Config.VAL_META_PATH = mock_val_meta_path
    Config.TEST_META_PATH = mock_test_meta_path

    # Update cache paths
    Config.VOCAB_PATH = os.path.join(cache_dir, "vocab.npy")
    Config.EMBEDDING_MATRIX_PATH = os.path.join(cache_dir, "embedding_matrix.npy")
    Config.TRAIN_FEATURES_PATH = os.path.join(cache_dir, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(cache_dir, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(cache_dir, "test_features.parquet")
    Config.MODEL_SAVE_PATH = os.path.join(base_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(base_dir, "submission.csv")

    # Speed optimizations
    Config.VOCAB_SIZE = 500
    Config.EMBED_DIM = 32
    Config.HIDDEN_DIM = 32
    Config.NUM_LAYERS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.DEBUG_SAMPLE_SIZE = None  # We already limited data via file creation
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data


def test_vocabulary_building():
    print("\n--- Testing Vocabulary Building ---")
    # Force build from scratch
    if os.path.exists(Config.VOCAB_PATH):
        os.remove(Config.VOCAB_PATH)

    vocab = get_vocab(load_cached_data=False)
    print(f"Vocabulary size: {len(vocab)}")

    # Assertions
    assert len(vocab) > 3, "Vocabulary should contain at least special tokens"
    assert Config.PAD_TOKEN in vocab.stoi, "PAD token missing"
    assert Config.UNK_TOKEN in vocab.stoi, "UNK token missing"

    # Test encoding
    text = "the quick brown fox"
    indices = vocab.encode(text, max_len=10)
    assert len(indices) == 10, "Encoding length mismatch"
    assert (
        indices[0] != vocab.stoi[Config.PAD_TOKEN]
    ), "First token shouldn't be PAD for valid text"

    return vocab


def test_model_architecture(vocab):
    print("\n--- Testing Model Architecture ---")
    device = torch.device("cpu")  # Use CPU for simple shape check

    model = CQCRNN(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=0.0,
    ).to(device)

    # Create dummy input
    batch_size = 2
    q_len = 10
    c_len = 50

    q_input = torch.randint(0, len(vocab), (batch_size, q_len))
    c_input = torch.randint(0, len(vocab), (batch_size, c_len))

    outputs = model(q_input, c_input)

    # Verify outputs
    print("Model output keys:", outputs.keys())
    assert "long_logits" in outputs
    assert "start_logits" in outputs
    assert "end_logits" in outputs
    assert "yn_logits" in outputs

    assert outputs["long_logits"].shape == (
        batch_size,
        1,
    ), f"Long logits shape mismatch: {outputs['long_logits'].shape}"
    assert outputs["start_logits"].shape == (
        batch_size,
        c_len,
    ), f"Start logits shape mismatch: {outputs['start_logits'].shape}"
    assert outputs["yn_logits"].shape == (
        batch_size,
        Config.NUM_CLASSES_YES_NO,
    ), f"YN logits shape mismatch: {outputs['yn_logits'].shape}"

    print("Model architecture verification passed.")


def test_training_pipeline():
    print("\n--- Testing Training Pipeline ---")

    # Run training using the engine
    # We set load_cached_data=False to ensure the pipeline runs fully
    try:
        train_model(
            epochs=Config.NUM_EPOCHS,
            patience=1,
            batch_size=Config.BATCH_SIZE,
            learning_rate=0.01,
            load_cached_data=False,
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        raise e

    # Verify model checkpoint exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise AssertionError(f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
    print(f"Model checkpoint verified at {Config.MODEL_SAVE_PATH}")


def test_inference_pipeline():
    print("\n--- Testing Inference Pipeline ---")

    # Ensure test offsets cache is cleared to force computation
    offsets_cache = os.path.join(Config.CACHE_DIR, "test_offsets.parquet")
    if os.path.exists(offsets_cache):
        os.remove(offsets_cache)

    try:
        run_inference_pipeline(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            load_cached_data=True,  # Use the features computed during training setup if available, or recompute
        )
        print("Inference completed successfully.")
    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_PATH}")

    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df.shape}")

    # Check format
    assert "example_id" in df.columns
    assert "PredictionString" in df.columns

    # We processed 10 test examples. Each has _long and _short rows. Total 20 rows.
    # Note: If collate_fn or dataset logic filters empty candidates, count might vary,
    # but based on logic it should output for every example_id.
    expected_rows = 10 * 2
    assert (
        len(df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df)}"

    print("Submission format verification passed.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()
    seed_everything(Config.SEED)

    # 2. Unit Tests
    vocab = test_vocabulary_building()
    test_model_architecture(vocab)

    # 3. Integration Tests (Train & Inference)
    test_training_pipeline()
    test_inference_pipeline()

    print("\nAll demonstrations and verifications passed successfully.")
