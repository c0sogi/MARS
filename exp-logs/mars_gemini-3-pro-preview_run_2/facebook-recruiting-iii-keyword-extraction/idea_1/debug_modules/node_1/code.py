import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library import config, utils, data, model, engine


def main():
    print("=== Starting Stack Exchange Tag Prediction Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # ---------------------------------------------------------
    print("\n[1] Configuring parameters for fast execution...")

    # Enable debug mode to use a small subset of data
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 2000  # Process only 2000 samples

    # Reduce training complexity
    config.NUM_EPOCHS = 1  # Train for only 1 epoch
    config.BATCH_SIZE = 32  # Small batch size
    config.LEARNING_RATE = 1e-3

    # Reduce model size for speed
    config.MAX_VOCAB_SIZE = 1000  # Smaller vocabulary
    config.NUM_TARGET_TAGS = 50  # Predict only top 50 tags
    config.MAX_SEQ_LEN = 100  # Shorter sequence length
    config.EMBEDDING_DIM = 64  # Smaller embeddings
    config.HIDDEN_DIMS = [64, 32]  # Smaller hidden layers

    # Update output paths to a dedicated demo directory to ensure a clean run
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    config.WORKING_DIR = demo_dir
    config.VOCAB_PATH = os.path.join(demo_dir, "vocab.json")
    config.TAG_MAP_PATH = os.path.join(demo_dir, "tag_map.json")
    config.TRAIN_PROCESSED_PATH = os.path.join(demo_dir, "train_processed.parquet")
    config.VAL_PROCESSED_PATH = os.path.join(demo_dir, "val_processed.parquet")
    config.TEST_PROCESSED_PATH = os.path.join(demo_dir, "test_processed.parquet")
    config.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_model.pth")
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Set random seed for reproducibility
    utils.set_seed(config.SEED)
    print("Configuration updated.")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("\n[2] Preparing DataLoaders...")

    # load_cached_data=False ensures we process the data with our new config parameters
    train_loader, val_loader, test_loader, vocab, tag_encoder = data.get_dataloaders(
        debug=config.DEBUG, load_cached_data=False
    )

    # Validation: Check DataLoaders
    print("Verifying DataLoaders...")
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Validation: Check Batch Structure
    batch = next(iter(train_loader))
    input_ids, lengths, labels, ids = batch

    print(f"Batch shapes -> Input: {input_ids.shape}, Labels: {labels.shape}")
    assert input_ids.dim() == 2, "Input IDs must be 2D (batch, seq_len)."
    assert labels.dim() == 2, "Labels must be 2D (batch, num_tags)."
    assert labels.shape[1] == len(
        tag_encoder
    ), f"Label size {labels.shape[1]} != Encoder size {len(tag_encoder)}."

    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Tag Encoder Size: {len(tag_encoder)}")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[3] Initializing DAN Model...")
    device = config.DEVICE
    dan_model = model.DAN(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        output_dim=len(tag_encoder),
        dropout_rate=config.DROPOUT_RATE,
        padding_idx=0,
    ).to(device)

    # Validation: Forward Pass
    print("Verifying Model Forward Pass...")
    input_ids = input_ids.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        logits = dan_model(input_ids, lengths)

    assert logits.shape == (
        input_ids.shape[0],
        len(tag_encoder),
    ), "Logits shape mismatch."
    print("Forward pass successful.")

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[4] Starting Training...")
    engine.train_model(dan_model, train_loader, val_loader, device)

    # Validation: Check if model artifact exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        # If early stopping didn't trigger save (unlikely in 1 epoch if val loss doesn't drop),
        # manually save for the next step.
        torch.save(dan_model.state_dict(), config.MODEL_SAVE_PATH)

    assert os.path.exists(config.MODEL_SAVE_PATH), "Model file was not saved."
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    print("\n[5] Generating Submission...")
    engine.generate_submission(dan_model, test_loader, tag_encoder, device)

    # Validation: Check submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    assert (
        "Id" in sub_df.columns and "Tags" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) > 0, "Submission file is empty."

    # Check if Tags are strings (even empty ones)
    assert sub_df["Tags"].dtype == object, "Tags column should be object/string type."
    print("Submission generated successfully.")

    # ---------------------------------------------------------
    # 6. Utility Function Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Utility Functions...")

    # Test F1 Score Calculation
    # Case: Perfect match
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[0, 1, 1], [1, 0, 0]])
    score = utils.calculate_f1_score(y_true, y_pred)
    assert score == 1.0, f"F1 Score failed. Expected 1.0, got {score}"

    # Case: No match
    y_pred_wrong = np.array([[1, 0, 0], [0, 1, 1]])
    score_wrong = utils.calculate_f1_score(y_true, y_pred_wrong)
    assert score_wrong == 0.0, f"F1 Score failed. Expected 0.0, got {score_wrong}"

    # Test Threshold Optimization
    # y_true: Class 1 is active.
    # y_probs: Class 1 has 0.4 prob.
    # If we threshold at 0.3, we predict 1 (Correct). If at 0.5, we predict 0 (Incorrect).
    y_true_opt = np.array([[0, 1]])
    y_probs_opt = np.array([[0.1, 0.4]])

    best_thresh, best_score = utils.optimize_threshold(
        y_true_opt, y_probs_opt, steps=10
    )

    # We expect the optimizer to find a threshold < 0.4 to capture the positive class
    assert best_score == 1.0, "Threshold optimization failed to maximize score."
    assert (
        best_thresh < 0.4
    ), f"Threshold optimization picked too high threshold: {best_thresh}"

    print("Utility functions verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
