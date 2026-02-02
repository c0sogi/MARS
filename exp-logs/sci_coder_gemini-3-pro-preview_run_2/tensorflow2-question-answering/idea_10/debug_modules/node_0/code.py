import os
import json
import torch
import numpy as np
import pandas as pd
import shutil
import random

# Import library components
from library.config import Config
from library.text_utils import build_or_load_tokenizer, build_or_load_embedding_matrix
from library.data_loader import get_data_loader
from library.network import SentenceFactorizedModel
from library.train_engine import TrainEngine
from library.predictor import Predictor


# --- Setup & Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# --- Configuration & Mock Data Setup ---
# We create a separate working directory for this demo to avoid conflicts
DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR)

print(f"--- Setting up Mock Data in {DEMO_DIR} ---")

# 1. Create Mock Raw Data (JSONL)
# Read top N lines from real files to create valid small datasets
SAMPLE_SIZE = 50

mock_train_path = os.path.join(DEMO_DIR, "mock_train.jsonl")
mock_test_path = os.path.join(DEMO_DIR, "mock_test.jsonl")


def create_mock_jsonl(source_path, dest_path, n):
    ids = []
    with open(source_path, "r", encoding="utf-8") as fin, open(
        dest_path, "w", encoding="utf-8"
    ) as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
            entry = json.loads(line)
            ids.append(str(entry["example_id"]))
    return ids


train_ids = create_mock_jsonl(Config.TRAIN_DATA_PATH, mock_train_path, SAMPLE_SIZE)
test_ids = create_mock_jsonl(Config.TEST_DATA_PATH, mock_test_path, SAMPLE_SIZE)

# 2. Create Mock Metadata (CSV)
# Split train_ids into train and val
split_idx = int(len(train_ids) * 0.8)
train_subset_ids = train_ids[:split_idx]
val_subset_ids = train_ids[split_idx:]

mock_train_meta = os.path.join(DEMO_DIR, "mock_train_meta.csv")
mock_val_meta = os.path.join(DEMO_DIR, "mock_val_meta.csv")
mock_test_meta = os.path.join(DEMO_DIR, "mock_test_meta.csv")

pd.DataFrame({"example_id": train_subset_ids}).to_csv(mock_train_meta, index=False)
pd.DataFrame({"example_id": val_subset_ids}).to_csv(mock_val_meta, index=False)
pd.DataFrame({"example_id": test_ids}).to_csv(mock_test_meta, index=False)

# 3. Override Config
print("--- Overriding Configuration for Demo ---")
Config.WORKING_DIR = DEMO_DIR
Config.TRAIN_DATA_PATH = mock_train_path
Config.TEST_DATA_PATH = mock_test_path
Config.TRAIN_META_PATH = mock_train_meta
Config.VAL_META_PATH = mock_val_meta
Config.TEST_META_PATH = mock_test_meta

# Update Cache Paths to point to demo dir
Config.VOCAB_PATH = os.path.join(DEMO_DIR, "vocab.npy")
Config.EMBEDDING_MATRIX_PATH = os.path.join(DEMO_DIR, "embedding_matrix.npy")
Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_data.parquet")
Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_data.parquet")
Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_data.parquet")
Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model.pth")
Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

# Reduce Hyperparameters for Speed
Config.MAX_VOCAB_SIZE = 1000
Config.EMBEDDING_DIM = 16
Config.HIDDEN_DIM = 32
Config.BATCH_SIZE = 4
Config.NUM_EPOCHS = 2
Config.DEBUG_SAMPLE_SIZE = None  # We already limited data via file creation

# --- Execution Pipeline ---

if __name__ == "__main__":

    # 1. Text Utilities: Tokenizer and Embeddings
    print("\n[1/5] Building Tokenizer and Embeddings...")
    tokenizer = build_or_load_tokenizer(load_cached_data=False)

    assert tokenizer.vocab_size > 0, "Tokenizer vocabulary is empty!"
    assert os.path.exists(Config.VOCAB_PATH), "Vocab file not saved!"

    embedding_matrix = build_or_load_embedding_matrix(tokenizer, load_cached_data=False)

    assert embedding_matrix.shape == (
        tokenizer.vocab_size,
        Config.EMBEDDING_DIM,
    ), f"Embedding shape mismatch: {embedding_matrix.shape}"

    # 2. Data Loading
    print("\n[2/5] Creating DataLoaders...")
    train_loader = get_data_loader(
        "train", tokenizer, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_data_loader(
        "val", tokenizer, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")

    assert "questions" in batch
    assert "sentences" in batch
    assert "labels" in batch
    assert "yes_no" in batch

    # Check dimensions
    # Questions: (batch_size, max_q_len)
    assert batch["questions"].shape[0] == Config.BATCH_SIZE
    # Yes/No: (batch_size)
    assert batch["yes_no"].shape[0] == Config.BATCH_SIZE
    # Sentences: (total_sentences_in_batch, max_sent_len) - variable dim 0
    assert batch["sentences"].dim() == 2
    assert batch["sentences"].shape[1] == Config.MAX_SENT_LEN

    print("Data Loader verification passed.")

    # 3. Model Initialization and Forward Pass
    print("\n[3/5] Initializing Model...")
    model = SentenceFactorizedModel(
        vocab_size=tokenizer.vocab_size, embedding_matrix=embedding_matrix
    )

    # Move batch to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    q_t = batch["questions"].to(device)
    s_t = batch["sentences"].to(device)
    dl_t = batch["doc_lengths"]  # list

    # Forward
    scores, yn_logits = model(q_t, s_t, dl_t)

    print(f"Scores shape: {scores.shape}")
    print(f"YN Logits shape: {yn_logits.shape}")

    # Assertions
    # Scores should match total number of sentences
    assert scores.shape[0] == s_t.shape[0]
    # YN Logits should match batch size and have 3 classes
    assert yn_logits.shape == (Config.BATCH_SIZE, 3)

    print("Model forward pass verification passed.")

    # 4. Training Loop
    print("\n[4/5] Running Training Engine...")
    engine = TrainEngine(model, device=device)

    # Train for defined epochs (2)
    engine.train(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved!"
    print("Training verification passed.")

    # 5. Prediction / Inference
    print("\n[5/5] Running Predictor...")
    # Initialize predictor (will reload tokenizer and model from checkpoint)
    predictor = Predictor(load_cached_data=True)

    # Generate submission
    predictor.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    # Check that we have rows for the test set
    # Each test example generates 2 rows (long and short)
    expected_rows = len(test_ids) * 2
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    print("Prediction verification passed.")
    print("\n--- All demonstrations completed successfully ---")
