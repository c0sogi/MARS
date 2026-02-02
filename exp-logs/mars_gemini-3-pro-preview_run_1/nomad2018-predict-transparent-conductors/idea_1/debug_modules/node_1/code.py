import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.architecture import LCDSModel
from library.engine import Engine


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Set random seed for reproducibility
    set_seed(42)

    # Modify Config parameters for a fast demonstration
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = (
        0  # Use main process for data loading to avoid overhead in demo
    )

    # Redirect output directories to ./working to ensure write permissions
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    Config.setup()
    print(f"Cache directory: {Config.CACHE_DIR}")
    print(f"Submission path: {Config.SUBMISSION_PATH}")

    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading datasets...")

    # get_dataloaders handles parsing XYZ files, feature extraction, and creating PyG DataLoaders.
    # We set load_cached_data=False to demonstrate the processing pipeline.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Validation: Check batch structure
    sample_batch = next(iter(train_loader))
    print("\nSample Batch Structure:")
    print(f"  x shape (atoms, features): {sample_batch.x.shape}")
    print(
        f"  lattice_features shape (graphs, features): {sample_batch.lattice_features.shape}"
    )
    print(f"  y shape (graphs, targets): {sample_batch.y.shape}")
    print(f"  batch index shape: {sample_batch.batch.shape}")

    # Assertions to ensure data is correct
    assert sample_batch.x.ndim == 2, "Node features should be 2D"
    assert sample_batch.lattice_features.ndim == 2, "Lattice features should be 2D"
    # Targets are log-transformed formation energy and bandgap energy
    assert sample_batch.y.shape[1] == 2, "Should have 2 target variables"

    # 3. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[3] Instantiating LCDS Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LCDSModel().to(device)

    # Validation: Perform a forward pass with the sample batch
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        output = model(sample_batch)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (sample_batch.num_graphs, 2), "Model output shape mismatch"
    print("Forward pass successful.")

    # 4. Training with Engine
    # -------------------------------------------------------------------------
    print("\n[4] Training Loop...")

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Engine
    engine = Engine(model, optimizer, device)

    # Run training
    # This uses the train_epoch and validate methods within Engine
    engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.NUM_EPOCHS,
        early_stopping_patience=1,
    )

    # Validation: Check if best model state was saved
    assert (
        engine.best_model_state is not None
    ), "Engine did not save the best model state."
    print("Training completed.")

    # 5. Prediction and Submission
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    # Use the engine to generate predictions on the test set
    # This handles evaluation mode, no_grad, and ID extraction
    engine.generate_submission(test_loader, Config.SUBMISSION_PATH)

    # Validation: Check if submission file exists and has correct format
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_submission.shape}")
    print("Head:")
    print(df_submission.head())

    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"
    assert len(df_submission) > 0, "Submission file is empty."

    # Check for valid values (energies should be non-negative after inverse transform)
    assert (
        df_submission["formation_energy_ev_natom"] >= 0
    ).all(), "Negative formation energy found"
    assert (
        df_submission["bandgap_energy_ev"] >= 0
    ).all(), "Negative bandgap energy found"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
