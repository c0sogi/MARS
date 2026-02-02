import os
import torch
import numpy as np
import pandas as pd
from library.config import Config, set_seed
from library.data_processing import get_dataloaders
from library.model import DAAN
from library.training import train_model
from library.inference import run_inference
from library.utils import compute_span_overlap_f1


def main():
    print("--- Starting DAAN Implementation Demonstration ---")

    # 1. Optimize Configuration for Speed
    print("Configuring hyperparameters for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 200  # Small subset for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.VOCAB_SIZE = 5000  # Smaller vocab for speed
    Config.EMBEDDING_DIM = 50  # Smaller embeddings
    Config.HIDDEN_DIM = 64

    # Ensure working directory is clean/ready (optional, handled by Config but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)

    # 2. Data Processing & Loading
    print("\n--- Testing Data Processing and Loading ---")
    # Note: load_cached_data=False forces regeneration of cache based on the new DEBUG config
    train_loader, val_loader, test_loader, embedding_matrix = get_dataloaders(
        load_cached_data=False
    )

    print(f"Embedding Matrix Shape: {embedding_matrix.shape}")
    assert (
        embedding_matrix.shape[1] == Config.EMBEDDING_DIM
    ), "Embedding dimension mismatch"

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    q_input = batch["q_input"]
    c_input = batch["c_input"]
    la_targets = batch["label_long"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Question Input Shape: {q_input.shape}")  # [Batch, MaxQ]
    print(f"Candidate Input Shape: {c_input.shape}")  # [Batch, MaxC]

    assert q_input.shape == (
        Config.BATCH_SIZE,
        Config.MAX_QUESTION_LEN,
    ), "Question input shape incorrect"
    assert c_input.shape == (
        Config.BATCH_SIZE,
        Config.MAX_CANDIDATE_LEN,
    ), "Candidate input shape incorrect"
    assert la_targets.shape == (
        Config.BATCH_SIZE,
    ), "Long answer target shape incorrect"

    # 3. Model Initialization and Forward Pass
    print("\n--- Testing Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAAN(embedding_matrix)
    model.to(device)

    q_input = q_input.to(device)
    c_input = c_input.to(device)

    la_logits, start_logits, end_logits = model(q_input, c_input)

    print(f"LA Logits Shape: {la_logits.shape}")
    print(f"Start Logits Shape: {start_logits.shape}")
    print(f"End Logits Shape: {end_logits.shape}")

    # Validation
    # la_logits: [Batch, 1]
    assert la_logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Long Answer logits shape incorrect"
    # start/end logits: [Batch, MaxC]
    assert start_logits.shape == (
        Config.BATCH_SIZE,
        Config.MAX_CANDIDATE_LEN,
    ), "Start logits shape incorrect"
    assert end_logits.shape == (
        Config.BATCH_SIZE,
        Config.MAX_CANDIDATE_LEN,
    ), "End logits shape incorrect"

    # 4. Utility Logic Verification
    print("\n--- Verifying Metrics Utility ---")
    # Test F1 calculation
    # Pred: (2, 4) -> indices {2, 3, 4}
    # True: [(2, 3)] -> indices {2, 3}
    # Intersection: {2, 3} (len 2)
    # Precision: 2/3, Recall: 2/2 = 1.0
    # F1: 2 * (2/3 * 1) / (2/3 + 1) = 1.33 / 1.66 = 0.8
    pred_span = (2, 4)
    true_spans = [(2, 3)]
    f1 = compute_span_overlap_f1(pred_span, true_spans)
    expected_f1 = 0.8
    print(f"Computed F1: {f1:.4f}, Expected: {expected_f1:.4f}")
    assert abs(f1 - expected_f1) < 1e-5, "F1 Score calculation logic error"

    # 5. Training Pipeline
    print("\n--- Executing Training Pipeline ---")
    # We use load_cached_data=True here because we just generated the cache in step 2
    train_model(load_cached_data=True)

    # Verify model checkpoint exists
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint successfully saved at {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created during training.")

    # 6. Inference Pipeline
    print("\n--- Executing Inference Pipeline ---")
    run_inference(load_cached_data=True)

    # Verify submission file exists
    if os.path.exists(Config.SUBMISSION_FILE):
        print(f"Submission file successfully created at {Config.SUBMISSION_FILE}")

        # Quick check of content
        df = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"Submission file shape: {df.shape}")
        print("First 5 rows:")
        print(df.head())

        # In debug mode with 200 samples, test set might be small, but should exist
        assert not df.empty, "Submission file is empty"
        assert (
            "example_id" in df.columns and "PredictionString" in df.columns
        ), "Submission columns missing"
    else:
        raise FileNotFoundError("Submission file was not created during inference.")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
