import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config, set_seed
from library.dataset import get_dataloader
from library.model import AudioMobileNet
from library.engine import (
    train_one_epoch,
    evaluate,
    calculate_lwlrap,
    generate_submission,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration of the Audio Classification pipeline...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    set_seed(42)

    # Modify Config for a fast demonstration (Speed Optimization)
    print("Configuring parameters for fast demonstration...")
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples per loader
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Check device
    device = Config.DEVICE
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying DataLoaders ---")

    # Initialize DataLoaders
    # We use debug=True to load only the subset defined in Config
    train_loader = get_dataloader("train", load_cached_data=False, debug=True)
    val_loader = get_dataloader("val", load_cached_data=False, debug=True)
    test_loader = get_dataloader("test", load_cached_data=False, debug=True)

    print(f"Train loader size (batches): {len(train_loader)}")

    # Fetch one batch to verify shapes
    inputs, targets = next(iter(train_loader))

    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    # Assertions
    # Expected Input: [Batch, 3, n_mels, time_steps]
    # Time steps calculation: 5s * 32000Hz / 320 hop ~ 500 frames (approx 501 due to padding)
    assert inputs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert (
        inputs.shape[1] == 3
    ), "Channel dimension mismatch (should be 3 for ImageNet model)"
    assert inputs.shape[2] == Config.N_MELS, "Mel band dimension mismatch"
    # Target: [Batch, 80 classes]
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"
    assert targets.shape[1] == Config.NUM_CLASSES, "Number of classes mismatch"

    print("DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model ---")

    model = AudioMobileNet(
        pretrained=False
    )  # False for speed, we don't need convergence
    model.to(device)

    # Move inputs to device
    inputs = inputs.to(device)

    # Forward pass
    outputs = model(inputs)

    print(f"Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Metric Verification (LWLRAP)
    # -------------------------------------------------------------------------
    print("\n--- Verifying LWLRAP Metric ---")

    # Create dummy data
    # Case 1: Perfect prediction
    truth_perfect = np.array([[1, 0, 1], [0, 1, 0]])
    scores_perfect = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]])

    lrap_perfect = calculate_lwlrap(truth_perfect, scores_perfect)
    print(f"Perfect Prediction LWLRAP: {lrap_perfect}")

    assert np.isclose(lrap_perfect, 1.0), "LWLRAP calculation failed for perfect case"

    # Case 2: Known imperfect prediction
    # Sample 1: Truth [1, 0], Score [0.4, 0.6] -> Rank 2 correct. Precision at rank 2 is 1/2.
    # LWLRAP = 0.5
    truth_simple = np.array([[1, 0]])
    scores_simple = np.array([[0.4, 0.6]])
    lrap_simple = calculate_lwlrap(truth_simple, scores_simple)

    assert np.isclose(
        lrap_simple, 0.5
    ), f"LWLRAP calculation failed for simple case. Got {lrap_simple}"

    print("Metric verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Verifying Training Loop ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")

    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # Evaluate
    val_loss, val_lrap = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation LRAP: {val_lrap:.4f}")

    assert 0 <= val_lrap <= 1.0, "Validation LRAP out of range [0, 1]"

    print("Training loop verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Verifying Submission Generation ---")

    submission_path = os.path.join(Config.WORK_DIR, "demo_submission.csv")

    # Generate submission using the test loader
    generate_submission(model, test_loader, device, submission_path)

    # Verify file creation
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify content format
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")

    # Check columns: fname + 80 classes
    expected_cols = 1 + Config.NUM_CLASSES
    assert (
        sub_df.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {sub_df.shape[1]}"

    # Check rows: generate_submission ensures the output matches the full sample_submission
    # length by filling missing predictions with 0s.
    ss_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    assert len(sub_df) == len(ss_df), f"Expected {len(ss_df)} rows, got {len(sub_df)}"

    print("Submission verification passed.")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
