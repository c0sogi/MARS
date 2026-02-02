import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.data_prep import process_data
from library.embeddings import get_embedding_matrix
from library.dataset import get_dataloaders
from library.model import IMCN
from library.trainer import train_model
from library.inference import predict


def run_demonstration():
    print("=== Starting Demonstration of IMCN Pipeline ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Debugging
    # ---------------------------------------------------------
    print("\n[1] Configuring hyperparameters for demo run...")
    # Reduce dataset size for speed
    Config.DEBUG_SAMPLE_SIZE = 100

    # Reduce model complexity
    Config.VOCAB_SIZE = 2000
    Config.EMBED_DIM = 32
    Config.HIDDEN_DIM = 32
    Config.NUM_FILTERS = 8

    # Reduce training duration
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16

    # Ensure clean state for working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.ensure_directories()

    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # ---------------------------------------------------------
    # 2. Data Preparation Verification
    # ---------------------------------------------------------
    print("\n[2] Running Data Preparation...")
    # Force processing (load_cached_data=False) to verify logic
    train_df, val_df, test_df, word2idx = process_data(load_cached_data=False)

    # Assertions to verify data integrity
    assert len(train_df) > 0, "Training dataframe is empty."
    assert len(val_df) > 0, "Validation dataframe is empty."
    assert len(test_df) > 0, "Test dataframe is empty."
    assert len(word2idx) > 0, "Vocabulary is empty."
    assert "q_indices" in train_df.columns, "Missing q_indices column."
    assert "c_indices" in train_df.columns, "Missing c_indices column."

    # Verify tokenization lengths
    sample_q = train_df.iloc[0]["q_indices"]
    sample_c = train_df.iloc[0]["c_indices"]
    assert (
        len(sample_q) == Config.MAX_Q_LEN
    ), f"Question length mismatch: {len(sample_q)} vs {Config.MAX_Q_LEN}"
    assert (
        len(sample_c) == Config.MAX_C_LEN
    ), f"Candidate length mismatch: {len(sample_c)} vs {Config.MAX_C_LEN}"

    print("Data preparation verified successfully.")

    # ---------------------------------------------------------
    # 3. Embeddings Verification
    # ---------------------------------------------------------
    print("\n[3] Generating Embeddings...")
    embedding_matrix = get_embedding_matrix(word2idx, load_cached_data=False)

    assert embedding_matrix.shape == (
        len(word2idx),
        Config.EMBED_DIM,
    ), f"Embedding shape mismatch: {embedding_matrix.shape}"

    print("Embedding matrix generated and verified.")

    # ---------------------------------------------------------
    # 4. DataLoader and Batch Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying DataLoaders...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify batch structure
    assert "q_indices" in batch
    assert "c_indices" in batch
    assert "label_long" in batch
    assert batch["q_indices"].shape == (Config.BATCH_SIZE, Config.MAX_Q_LEN)
    assert batch["c_indices"].shape == (Config.BATCH_SIZE, Config.MAX_C_LEN)
    assert batch["label_long"].shape == (Config.BATCH_SIZE,)

    print("DataLoader batch structure verified.")

    # ---------------------------------------------------------
    # 5. Model Instantiation and Forward Pass
    # ---------------------------------------------------------
    print("\n[5] Initializing Model and Testing Forward Pass...")
    model = IMCN(embedding_matrix)

    # Move batch to CPU for this test (model is on CPU by default)
    q_in = batch["q_indices"]
    c_in = batch["c_indices"]

    la_logits, start_logits, end_logits = model(q_in, c_in)

    # Verify output shapes
    assert la_logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"LA Logits shape mismatch: {la_logits.shape}"
    assert start_logits.shape == (
        Config.BATCH_SIZE,
        Config.MAX_C_LEN,
    ), f"Start Logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        Config.BATCH_SIZE,
        Config.MAX_C_LEN,
    ), f"End Logits shape mismatch: {end_logits.shape}"

    print("Model forward pass verified.")

    # ---------------------------------------------------------
    # 6. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop...")
    # This function handles device movement internally
    train_model(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
    )

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print("Training loop completed and model saved.")

    # ---------------------------------------------------------
    # 7. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[7] Running Inference...")
    predict(load_cached_data=True, batch_size=Config.BATCH_SIZE)

    # Verify submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, Config.SUBMISSION_FILE)
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Check submission content format
    sub_df = pd.read_csv(submission_path)
    assert "example_id" in sub_df.columns
    assert "PredictionString" in sub_df.columns
    assert len(sub_df) > 0

    print(f"Submission generated with {len(sub_df)} rows.")
    print("Sample rows:")
    print(sub_df.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
