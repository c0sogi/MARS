import os
import torch
import numpy as np
import pandas as pd
from library.config import Config, set_seed
from library.utils import compute_levenshtein, decode_predictions
from library.data_loader import get_dataloaders
from library.model import BiLSTMClassifier
from library.trainer import run_training
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast debug session
    print("\n[1] Configuring environment for fast execution...")
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples per split
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.PATIENCE = 1

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: 2 Epochs, Batch Size 4, Subset Size 10.")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test Levenshtein Distance
    # Truth: [1, 2, 3], Pred: [1, 2] -> Distance 1 (Deletion)
    # Truth: [1, 2], Pred: [1, 3] -> Distance 1 (Substitution)
    # Total Distance = 2, Total Truth Length = 5 -> Error = 0.4
    truth_seqs = [[1, 2, 3], [1, 2]]
    pred_seqs = [[1, 2], [1, 3]]
    error_rate = compute_levenshtein(pred_seqs, truth_seqs)
    expected_error = 2.0 / 5.0
    assert (
        abs(error_rate - expected_error) < 1e-6
    ), f"Levenshtein calculation failed. Expected {expected_error}, got {error_rate}"
    print("Levenshtein metric verification passed.")

    # Test Decode Predictions
    # Logits: (1, 3, 2) -> Batch=1, Time=3, Classes=2 (Background=0, Gesture=1)
    # Frame 0: Class 1, Frame 1: Class 1, Frame 2: Class 0
    # Expected: [1] (Collapse repeats 1->1, remove background 0)
    dummy_logits = torch.tensor(
        [[[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]]]
    )  # Shape (1, 3, 2)
    decoded = decode_predictions(dummy_logits)
    assert decoded == [[1]], f"Decoding failed. Expected [[1]], got {decoded}"
    print("Prediction decoding verification passed.")

    # 3. Data Loading
    print("\n[3] Loading Data (Debug Mode)...")
    # This will process a small subset of MAT/Audio files and cache them
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Verify Batch Structure
    batch = next(iter(train_loader))
    features = batch["features"]
    labels = batch["labels"]
    lengths = batch["lengths"]
    ids = batch["sample_ids"]

    print(
        f"Batch shapes - Features: {features.shape}, Labels: {labels.shape}, Lengths: {lengths.shape}"
    )

    # Assertions
    assert features.dim() == 3, "Features should be 3D (Batch, Time, Dim)"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Feature dim should be {Config.INPUT_DIM}"
    assert labels.dim() == 2, "Labels should be 2D (Batch, Time)"
    assert len(ids) == features.shape[0], "Mismatch in batch size and sample IDs"
    print("Data Loader verification passed.")

    # 4. Model Initialization & Forward Pass
    print("\n[4] Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = BiLSTMClassifier().to(device)

    # Move batch to device
    features = features.to(device)
    lengths = lengths.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(features, lengths)

    print(f"Output Logits Shape: {logits.shape}")
    assert logits.shape == (
        features.shape[0],
        features.shape[1],
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("Model forward pass verification passed.")

    # 5. Training Loop
    print("\n[5] Running Training Loop...")
    # This uses the Trainer class internally
    trainer = run_training(model, train_loader, val_loader)

    # Check if checkpoint was saved
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not created."
    print(f"Training complete. Checkpoint saved at {Config.MODEL_CHECKPOINT}")

    # 6. Inference
    print("\n[6] Running Inference...")
    # This uses the Predictor class internally to generate submission.csv
    run_inference(
        checkpoint_path=Config.MODEL_CHECKPOINT,
        output_path=Config.SUBMISSION_FILE,
        debug=True,
    )

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Validate Submission Format
    print("Validating submission format...")
    with open(Config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()

    if lines:
        # Check first row
        parts = lines[0].strip().split(",")
        first_id = parts[0]
        first_labels = ",".join(parts[1:])

        print(f"Sample Submission Row: {first_id} -> {first_labels}")
        assert (
            isinstance(first_id, str) and len(first_id) > 0
        ), "SessionID should be a string"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
