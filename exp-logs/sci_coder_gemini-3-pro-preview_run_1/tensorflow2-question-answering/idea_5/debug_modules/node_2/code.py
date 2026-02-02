import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.text_processing import build_vocab, create_embedding_matrix
from library.data_loader import process_split, NQDataset, collate_fn
from library.model import get_model, QCBiGRU
from library.engine import Engine
from library.predictor import generate_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline_demonstration():
    print("=== Starting Pipeline Demonstration ===")

    # 1. Configuration for Speed
    print("\n[Demo] Configuring for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.MAX_VOCAB_SIZE = 1000  # Small vocab for speed
    Config.EMBEDDING_DIM = 32  # Small embedding dim
    Config.HIDDEN_SIZE = 32

    # Ensure working directory is clean-ish or ready
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    set_seed(Config.SEED)

    # 2. Text Processing (Vocab & Embeddings)
    print("\n[Demo] Building Vocabulary and Embeddings...")
    # Force rebuild to demonstrate logic
    vocab = build_vocab(load_cached_data=False)

    assert len(vocab) > 0, "Vocabulary should not be empty"
    assert Config.PAD_TOKEN in vocab, "PAD token missing from vocab"
    assert Config.UNK_TOKEN in vocab, "UNK token missing from vocab"

    embedding_matrix = create_embedding_matrix(vocab, load_cached_data=False)

    assert embedding_matrix.shape == (
        len(vocab),
        Config.EMBEDDING_DIM,
    ), f"Embedding shape mismatch. Expected {(len(vocab), Config.EMBEDDING_DIM)}, got {embedding_matrix.shape}"

    print("[Demo] Text processing artifacts verified.")

    # 3. Data Loading
    print("\n[Demo] Processing Data Splits...")
    # Process train and val splits
    train_df = process_split("train", load_cached_data=False)
    val_df = process_split("val", load_cached_data=False)

    assert not train_df.empty, "Training dataframe is empty"
    assert not val_df.empty, "Validation dataframe is empty"

    # Check columns
    required_cols = ["example_id", "question_text", "candidate_text", "long_label"]
    for col in required_cols:
        assert col in train_df.columns, f"Missing column {col} in processed data"

    # Create Datasets and Loaders
    train_dataset = NQDataset(train_df, vocab)
    val_dataset = NQDataset(val_df, vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    print(f"[Demo] Batch keys: {batch.keys()}")
    assert "question_ids" in batch
    assert "candidate_ids" in batch
    assert batch["question_ids"].shape[0] == Config.BATCH_SIZE or batch[
        "question_ids"
    ].shape[0] == len(train_df)

    print(f"[Demo] Data loading verified. Train samples: {len(train_dataset)}")

    # 4. Model Initialization
    print("\n[Demo] Initializing Model...")
    # Force get_model to use the embedding matrix we just created (it loads from disk path defined in Config)
    model = get_model(load_weights=False)
    model.to(Config.DEVICE)

    assert isinstance(model, QCBiGRU), "Model is not instance of QCBiGRU"

    # Test Forward Pass
    q_ids = batch["question_ids"].to(Config.DEVICE)
    c_ids = batch["candidate_ids"].to(Config.DEVICE)

    long_prob, start_logits, end_logits = model(q_ids, c_ids)

    assert long_prob.shape == (
        q_ids.size(0),
        1,
    ), f"Long prob shape mismatch: {long_prob.shape}"
    assert (
        start_logits.shape == c_ids.shape
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert (
        end_logits.shape == c_ids.shape
    ), f"End logits shape mismatch: {end_logits.shape}"

    print("[Demo] Model forward pass verified.")

    # 5. Training (Engine)
    print("\n[Demo] Running Training Loop...")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    engine = Engine(model, Config.DEVICE, optimizer)

    # Run fit (train + eval + save)
    engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=Config.MODEL_SAVE_PATH,
    )

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("[Demo] Training loop completed and model saved.")

    # 6. Inference (Predictor)
    print("\n[Demo] Generating Predictions...")
    # This function loads the saved model and processes the test set (using DEBUG size)
    generate_predictions()

    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"[Demo] Submission loaded. Rows: {len(sub_df)}")
    assert "example_id" in sub_df.columns
    assert "PredictionString" in sub_df.columns

    # Verify format of example_id (should end in _long or _short)
    if len(sub_df) > 0:
        ex_id = sub_df.iloc[0]["example_id"]
        assert ex_id.endswith("_long") or ex_id.endswith(
            "_short"
        ), f"Invalid example_id format: {ex_id}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demonstration()
