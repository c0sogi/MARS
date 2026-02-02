import torch
import numpy as np
import os
import sys

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, GaussianRBF, TargetScaler
from library.data import get_dataloaders
from library.model import EUGAT
from library.train import run_training


def demo_utils():
    """
    Demonstrates and verifies the utility classes: GaussianRBF and TargetScaler.
    """
    print("--- Demonstrating Utils ---")

    # 1. GaussianRBF
    # Create dummy distances: 5 edges with distances between 0.5 and 5.0
    distances = torch.tensor([0.5, 1.0, 2.5, 4.0, 5.0])
    # Instantiate RBF with 10 centers
    rbf = GaussianRBF(start=0.0, stop=5.0, n_centers=10)
    # Forward pass
    rbf_features = rbf(distances)

    print(f"GaussianRBF input shape: {distances.shape}")
    print(f"GaussianRBF output shape: {rbf_features.shape}")

    # Verification
    assert rbf_features.shape == (5, 10), "GaussianRBF output shape mismatch."
    # Gaussian values should be between 0 and 1
    assert (
        rbf_features.min() >= 0 and rbf_features.max() <= 1
    ), "RBF values out of range [0, 1]."
    print("GaussianRBF logic verified.")

    # 2. TargetScaler
    # Create dummy targets: 5 samples, 2 targets
    targets = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]
    )
    scaler = TargetScaler()
    scaler.fit(targets)

    # Transform
    transformed = scaler.transform(targets)
    # Inverse transform
    reconstructed = scaler.inverse_transform(transformed)

    print(f"Original targets mean: {targets.mean(dim=0)}")
    print(f"Transformed targets mean: {transformed.mean(dim=0)} (Should be close to 0)")

    # Verification
    # Transformed data should have zero mean
    assert torch.allclose(
        transformed.mean(dim=0), torch.zeros(2), atol=1e-6
    ), "Scaler mean not zero."
    # Reconstructed data should match original
    assert torch.allclose(
        targets, reconstructed, atol=1e-5
    ), "Scaler reconstruction failed."
    print("TargetScaler logic verified.")
    print()


def demo_data_and_model():
    """
    Demonstrates data loading and model instantiation/forward pass.
    """
    print("--- Demonstrating Data Loading and Model ---")

    # 1. Data Loading
    # We use a small batch size for demonstration
    batch_size = 4
    print(f"Initializing DataLoaders with batch_size={batch_size}...")

    # This will trigger processing if cache is missing
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Fetch a single batch from the training loader
    batch = next(iter(train_loader))
    print(f"Fetched batch: {batch}")
    print(f"  - num_graphs: {batch.num_graphs}")
    print(f"  - x (atomic numbers) shape: {batch.x.shape}")
    print(f"  - edge_index shape: {batch.edge_index.shape}")
    print(f"  - edge_attr (distances) shape: {batch.edge_attr.shape}")
    print(f"  - y (targets) shape: {batch.y.shape}")

    # Verification of batch structure
    assert batch.x.dim() == 1, "Node features should be 1D (atomic numbers)."
    assert batch.edge_index.shape[0] == 2, "Edge index should have 2 rows."
    assert (
        batch.edge_attr.ndim == 2 and batch.edge_attr.shape[1] == 1
    ), "Edge attr should be [E, 1]."
    assert batch.y.shape[1] == 2, "Targets should have 2 columns."
    print("Data loading verified.")

    # 2. Model Initialization and Forward Pass
    print("Initializing EUGAT model...")
    model = EUGAT()

    # Forward pass with the fetched batch
    # The model handles node embedding, edge expansion, message passing, and pooling internally
    output = model(batch)

    print(f"Model output shape: {output.shape}")

    # Verification of output shape [batch_size, num_targets]
    assert output.shape == (
        batch_size,
        2,
    ), f"Expected output shape ({batch_size}, 2), got {output.shape}"
    print("Model forward pass verified.")
    print()


def demo_training_pipeline():
    """
    Demonstrates the full training pipeline using the run_training function.
    """
    print("--- Demonstrating Training Pipeline ---")

    # We run the training function provided in the library.
    # We limit epochs to 1 to keep execution time short for demonstration purposes.
    # This function handles:
    # - Data loading
    # - Model initialization
    # - Optimizer/Scheduler setup
    # - Training loop
    # - Validation
    # - Checkpointing
    # - Submission generation

    print("Running training for 1 epoch...")
    run_training(
        num_epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        load_cached_data=True,  # Will reuse cache generated in previous step
        patience=1,
    )

    # Verification of artifacts
    submission_path = Config.SUBMISSION_PATH
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(submission_path):
        print(f"Verification: Submission file found at {submission_path}")
    else:
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    if os.path.exists(checkpoint_path):
        print(f"Verification: Checkpoint file found at {checkpoint_path}")
    else:
        print(
            f"Warning: Checkpoint file not found at {checkpoint_path}. (Val loss might not have improved)"
        )

    print("Training pipeline demonstration finished.")


if __name__ == "__main__":
    # Set global seed for reproducibility
    set_seed(42)

    try:
        demo_utils()
        demo_data_and_model()
        demo_training_pipeline()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        raise e
