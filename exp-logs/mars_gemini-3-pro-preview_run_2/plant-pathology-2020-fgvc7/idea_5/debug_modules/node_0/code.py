import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import AppleDataset, get_transforms
from library.model import AppleDiseaseModel
from library.engine import train_one_epoch, evaluate, reconstruct_probabilities


def run_demonstration():
    print("=== Starting Apple Disease Detection Library Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.IMG_SIZE = 224
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print(f"    Image Size: {Config.IMG_SIZE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset and Data Loading...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Sample a small subset for demo (10 samples)
    subset_df = full_train_df.head(10).copy()
    print(f"    Created subset of {len(subset_df)} samples from training metadata.")

    # Initialize Dataset
    # We use 'resnet18' later, so transforms resizing to 224 is appropriate
    train_dataset = AppleDataset(
        subset_df, transforms=get_transforms("train"), mode="train"
    )

    # Verify __getitem__
    sample_img, sample_target = train_dataset[0]

    # Check Image Shape: (3, H, W)
    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {sample_img.shape}"

    # Check Target Shape: (2,) -> [Rust, Scab]
    assert sample_target.shape == (
        2,
    ), f"Target shape mismatch. Expected (2,), got {sample_target.shape}"

    print("    Dataset __getitem__ verification passed.")
    print(f"    Sample Target (Rust, Scab): {sample_target}")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Initializing Model and Verifying Forward Pass...")

    # Use a lightweight model for the demo instead of EfficientNet-B5
    # We pass 'resnet18' to the constructor.
    model = AppleDiseaseModel(model_name="resnet18", pretrained=False)
    model.to(Config.DEVICE)

    # Create dummy input batch
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    # Verify Output Shape: (Batch, 2)
    assert logits.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 2), got {logits.shape}"

    print("    Model forward pass successful. Output shape verified.")

    # ---------------------------------------------------------
    # 4. Logic Verification: Probability Reconstruction
    # ---------------------------------------------------------
    print("\n[4] Verifying Probability Reconstruction Logic...")

    # Create synthetic probabilities for [Rust, Scab]
    # Case 1: High Rust, Low Scab -> Should result in high 'Rust' class
    # Case 2: High Rust, High Scab -> Should result in high 'Multiple' class
    input_probs = np.array(
        [[0.9, 0.1], [0.8, 0.8]]  # Rust=0.9, Scab=0.1  # Rust=0.8, Scab=0.8
    )

    reconstructed = reconstruct_probabilities(input_probs)
    # Expected columns: [Healthy, Multiple, Rust, Scab]

    # Check Case 1: Rust (Index 2) should be dominant
    # Rust = pr * (1 - ps) = 0.9 * 0.9 = 0.81
    assert np.isclose(
        reconstructed[0, 2], 0.81
    ), f"Reconstruction logic error for Rust class. Got {reconstructed[0, 2]}"

    # Check Case 2: Multiple (Index 1) should be dominant
    # Multiple = pr * ps = 0.8 * 0.8 = 0.64
    assert np.isclose(
        reconstructed[1, 1], 0.64
    ), f"Reconstruction logic error for Multiple class. Got {reconstructed[1, 1]}"

    # Check Sum to 1
    row_sums = reconstructed.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0
    ), f"Probabilities do not sum to 1. Sums: {row_sums}"

    print("    Probability reconstruction logic verified.")

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Training Step...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch (on the tiny subset)
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, Config.DEVICE
    )

    print(f"    Training step complete. Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # ---------------------------------------------------------
    # 6. Evaluation Demonstration
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Evaluation Step...")

    # Use the same loader as validation for demo purposes
    val_loss, val_auc = evaluate(model, train_loader, criterion, Config.DEVICE)

    print(f"    Evaluation complete. Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # ---------------------------------------------------------
    # 7. Inference Output Formatting
    # ---------------------------------------------------------
    print("\n[7] Simulating Inference Output...")

    # Simulate logits from the model
    simulated_logits = torch.tensor([[2.0, -2.0], [-1.0, 1.5]])  # Batch of 2
    simulated_probs_2d = torch.sigmoid(simulated_logits).numpy()

    final_probs = reconstruct_probabilities(simulated_probs_2d)

    # Create submission entries
    demo_ids = ["Test_Demo_1", "Test_Demo_2"]
    results = []
    for i, img_id in enumerate(demo_ids):
        results.append(
            {
                "image_id": img_id,
                "healthy": final_probs[i, 0],
                "multiple_diseases": final_probs[i, 1],
                "rust": final_probs[i, 2],
                "scab": final_probs[i, 3],
            }
        )

    submission_df = pd.DataFrame(results)
    print("    Generated Submission DataFrame:")
    print(submission_df.to_string(index=False))

    # Verify columns
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
