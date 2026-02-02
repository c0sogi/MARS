import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil

# Import library modules
from library.config import Config
from library.utils import set_seeds, compute_levenshtein_distance
from library.data_loader import get_dataloaders
from library.model import BiGRUModel
from library.trainer import Trainer
from library.inference import run_inference


def main():
    # ==========================================
    # 1. Setup and Config Override for Demo
    # ==========================================
    # Set seeds for reproducibility
    set_seeds(42)

    print("Setting up demonstration environment...")

    # Define paths for mini-dataset
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Load original metadata and create a small subset (top 10 samples)
    # This ensures the demo runs quickly without processing all videos
    try:
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        train_df.head(10).to_csv(mini_train_path, index=False)
        val_df.head(5).to_csv(mini_val_path, index=False)
        test_df.head(5).to_csv(mini_test_path, index=False)
        print("Mini-dataset created successfully.")
    except FileNotFoundError as e:
        print(f"Error reading metadata: {e}")
        sys.exit(1)

    # Runtime Override of Config parameters
    # We modify the class attributes directly so they propagate to other modules
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # Use .npz extension to ensure compatibility with np.savez/np.load in data_loader
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "mini_train.npz")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "mini_val.npz")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "mini_test.npz")

    # Reduced hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.HIDDEN_DIM = 32
    Config.BATCH_SIZE = 4
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.USE_CLASS_WEIGHTS = False  # Simplify for demo

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n--- Testing Data Loader ---")
    # load_cached_data=False forces processing of the new mini-dataset
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch to verify shapes
    try:
        features, labels, lengths = next(iter(train_loader))
        print(
            f"Batch Shapes -> Features: {features.shape}, Labels: {labels.shape}, Lengths: {lengths.shape}"
        )

        # Assertions
        # Features: (Batch, MaxLen, InputDim)
        assert features.ndim == 3, "Features should be 3D tensor"
        assert (
            features.shape[2] == Config.INPUT_DIM
        ), f"Feature dim should be {Config.INPUT_DIM}"
        # Labels: (Batch, MaxLen)
        assert labels.ndim == 2, "Labels should be 2D tensor"
        assert (
            lengths.shape[0] == Config.BATCH_SIZE
            or lengths.shape[0] == features.shape[0]
        ), "Lengths should match batch size"
        print("Data Loader assertions passed.")
    except StopIteration:
        print("Error: Train loader is empty.")
        sys.exit(1)

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n--- Testing Model ---")
    model = BiGRUModel(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    )

    # Move to appropriate device (CPU for demo is fine, but code uses CUDA if avail)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features = features.to(device)
    # Lengths usually stay on CPU for packing or are handled inside model

    with torch.no_grad():
        logits = model(features, lengths)

    print(f"Logits Shape: {logits.shape}")
    # Assertions
    assert logits.shape[0] == features.shape[0], "Batch size mismatch in output"
    assert logits.shape[1] == features.shape[1], "Sequence length mismatch in output"
    assert logits.shape[2] == Config.NUM_CLASSES, "Class dimension mismatch"
    print("Model forward pass assertions passed.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n--- Testing Trainer ---")
    trainer = Trainer(model, train_loader, val_loader, config=Config)

    # Run training
    trainer.train()

    # Assertions
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print("Training finished and model saved.")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("\n--- Testing Inference ---")
    # Run inference using the library function
    # It will use the Config paths we overrode
    run_inference(load_cached_data=False)

    # Assertions
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not generated"

    # Verify submission content
    print(f"Reading submission file: {Config.SUBMISSION_FILE}")
    with open(Config.SUBMISSION_FILE, "r") as f:
        # Read non-empty lines
        submission_lines = [line.strip() for line in f.readlines() if line.strip()]

    # Parse rows by splitting on comma
    submission_rows = [line.split(",") for line in submission_lines]

    print(f"Submission generated with {len(submission_rows)} rows.")

    # Check if number of rows matches test set
    # Note: inference writes all processed samples.
    # Our mini test set has 5 samples.
    assert (
        len(submission_rows) == 5
    ), f"Expected 5 predictions, got {len(submission_rows)}"

    # Check first ID
    # Row 0, Col 0 should be the SessionID
    expected_id = pd.read_csv(mini_test_path).iloc[0]["sample_id"]
    actual_id = submission_rows[0][0]
    assert (
        expected_id == actual_id
    ), f"ID mismatch: Expected {expected_id}, got {actual_id}"
    print("Inference assertions passed.")

    # ==========================================
    # 6. Metric Demonstration
    # ==========================================
    print("\n--- Testing Metric ---")
    # Synthetic Ground Truth: [Class 1, Class 2]
    # Synthetic Prediction: [Class 1, Class 3] (1 substitution)
    gt = [[1, 2]]
    preds = [[1, 3]]

    # Levenshtein distance between [1,2] and [1,3] is 1 (substitution of 2->3)
    # Total length of GT is 2.
    # Metric = 1 / 2 = 0.5

    score = compute_levenshtein_distance(gt, preds)
    print(f"Calculated Score: {score}")
    assert abs(score - 0.5) < 1e-6, "Metric calculation incorrect"
    print("Metric assertions passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
