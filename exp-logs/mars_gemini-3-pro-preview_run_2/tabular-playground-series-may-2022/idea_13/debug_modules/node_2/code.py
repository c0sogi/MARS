import os
import sys
import torch
import pandas as pd
import numpy as np

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.utils import set_seed, get_device
from library.model import DualViewHybridResFunnel
from library.train import run_training


def demo_model_instantiation():
    """
    Demonstrates how to instantiate the model and verifies a forward pass
    with dummy data.
    """
    print("\n--- Demonstrating Model Instantiation and Forward Pass ---")
    device = get_device()
    print(f"Device: {device}")

    # Instantiate model with specific hyperparameters
    # We use smaller dimensions here just for the structural check
    model = DualViewHybridResFunnel(
        num_continuous=30,
        vocab_size=32,  # Covers A-Z (indices 1-26) + padding (0)
        embedding_dim=16,
        seq_len=10,
        transformer_layers=1,
        backbone_dims=[64, 32],
        dropout=0.1,
    ).to(device)

    # Create dummy input data
    # Continuous features: (Batch, 30)
    batch_size = 8
    x_cont = torch.randn(batch_size, 30).to(device)

    # Categorical features: (Batch, 10), values between 1 and 31
    x_cat = torch.randint(1, 32, (batch_size, 10)).to(device)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        output = model(x_cont, x_cat)

    # Validate output
    # Output should be (Batch, 1) and values in [0, 1] due to Sigmoid
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected shape {(batch_size, 1)}, got {output.shape}"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Output probabilities must be in [0, 1]"

    print("Model instantiated and forward pass verified successfully.")


def demo_training_pipeline():
    """
    Demonstrates the full training pipeline using the library's run_training function.
    Optimized for speed by running only 1 epoch with a large batch size.
    """
    print("\n--- Demonstrating Full Training Pipeline ---")

    # Configuration
    WORK_DIR = "./working/demo_execution"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # Execute the training pipeline
    # This handles data loading (with caching), model init, training loop, and inference.
    # We set epochs=1 and a large batch size to complete the demo in seconds.
    print("Starting run_training (Epochs: 1, Batch Size: 4096)...")
    run_training(
        epochs=1,
        batch_size=4096,
        learning_rate=1e-3,
        weight_decay=1e-2,
        work_dir=WORK_DIR,
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
    )

    # Verify the artifacts were created
    submission_path = os.path.join(WORK_DIR, "submission.csv")
    model_path = os.path.join(WORK_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file was not saved at {model_path}")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Submission file was not generated at {submission_path}"
        )

    # Verify submission integrity
    df_sub = pd.read_csv(submission_path)
    expected_len = 100000

    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch: {len(df_sub)} != {expected_len}"
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "target" in df_sub.columns, "Submission missing 'target' column"

    print(f"Pipeline executed successfully. Files generated in {WORK_DIR}")


if __name__ == "__main__":
    # Set fixed seed for reproducibility across the entire script
    set_seed(42)

    try:
        # 1. Verify Model Component
        demo_model_instantiation()

        # 2. Verify Full Pipeline
        demo_training_pipeline()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
