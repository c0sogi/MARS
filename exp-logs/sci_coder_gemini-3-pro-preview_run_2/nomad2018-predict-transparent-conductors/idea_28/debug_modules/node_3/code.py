import sys
import os
import torch
import numpy as np
import pandas as pd

# Add current directory to sys.path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import get_scaler, compute_metric, TargetScaler
from library.data import get_dataloaders
from library.model import SRACGN, generate_submission
from library.train import run_training, set_seed


def main():
    print("Starting demonstration script...")

    # -------------------------------------------------------------------------
    # 1. Configure for Demo (Speed Optimization)
    # -------------------------------------------------------------------------
    print("\n--- Configuring for Demo ---")
    # Enable DEBUG mode to use only the first 50 samples of train/val/test data
    Config.DEBUG = True

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 8

    # Redirect outputs to a demo directory to avoid overwriting production work
    Config.WORKING_DIR = "./working/demo_run"
    Config.TRAIN_GRAPH_CACHE = os.path.join(
        Config.WORKING_DIR, "cache", "train_graphs.npz"
    )
    Config.VAL_GRAPH_CACHE = os.path.join(Config.WORKING_DIR, "cache", "val_graphs.npz")
    Config.TEST_GRAPH_CACHE = os.path.join(
        Config.WORKING_DIR, "cache", "test_graphs.npz"
    )
    Config.TARGET_SCALER_CACHE = os.path.join(
        Config.WORKING_DIR, "cache", "target_scaler.npz"
    )
    Config.MODEL_CHECKPOINT_PATH = os.path.join(
        Config.WORKING_DIR, "checkpoints", "best_model.pth"
    )
    Config.SUBMISSION_PATH = os.path.join("./working/demo_submission", "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(Config.TRAIN_GRAPH_CACHE), exist_ok=True)
    os.makedirs(os.path.dirname(Config.MODEL_CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for fast execution.")

    # -------------------------------------------------------------------------
    # 2. Test Utilities
    # -------------------------------------------------------------------------
    print("\n--- Testing Utilities ---")

    # Test Metric Calculation
    y_true = np.array([[1.0, 2.0], [0.5, 0.5]])
    y_pred = np.array([[1.1, 1.9], [0.5, 0.6]])
    metric = compute_metric(y_true, y_pred)
    print(f"Computed RMSLE: {metric:.6f}")
    assert metric >= 0, "Metric should be non-negative"

    # Test TargetScaler
    scaler = TargetScaler()
    dummy_data = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    scaler.fit(dummy_data)
    transformed = scaler.transform(dummy_data)
    inverse = scaler.inverse_transform(transformed)

    print("Scaler verification:")
    print(f"  Original: {dummy_data[0]}")
    print(f"  Inverse:  {inverse[0]}")

    assert np.allclose(
        dummy_data, inverse, atol=1e-5
    ), "Scaler inverse transform failed to recover original data."
    print("Utilities verified.")

    # -------------------------------------------------------------------------
    # 3. Test Data Processing and Loading
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Processing ---")
    # Force reload_cached_data=False to demonstrate processing from scratch
    # In DEBUG mode, this processes only 50 samples per split
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Verify batch structure
    batch = next(iter(train_loader))
    print("\nSample Batch Structure:")
    print(batch)

    assert hasattr(batch, "x"), "Batch missing node features 'x'"
    assert hasattr(batch, "edge_index"), "Batch missing 'edge_index'"
    assert hasattr(batch, "edge_attr"), "Batch missing 'edge_attr'"
    assert hasattr(batch, "y"), "Batch missing targets 'y'"

    # Check dimensions
    assert batch.x.ndim == 1, "Node features should be 1D (atomic numbers)"
    assert batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert batch.y.shape[1] == 2, "Targets should have 2 columns"
    print("Data processing and loading verified.")

    # -------------------------------------------------------------------------
    # 4. Test Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Forward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SRACGN().to(device)
    batch = batch.to(device)

    with torch.no_grad():
        output = model(batch)

    print(f"Model Output Shape: {output.shape}")
    expected_shape = (batch.num_graphs, 2)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Test Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Testing Training Loop ---")
    # Run training using the library function
    # This will train for Config.NUM_EPOCHS (2) and save the best model
    run_training(
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        load_cached_data=True,  # Use the cache we just generated
    )

    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not created."
    print("Training loop execution verified.")

    # -------------------------------------------------------------------------
    # 6. Test Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Testing Submission Generation ---")
    # This function loads the best model and the test set, then generates predictions
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("\nSubmission File Head:")
    print(sub_df.head())

    assert len(sub_df) > 0, "Submission file is empty"
    assert "id" in sub_df.columns, "Submission missing 'id' column"
    assert (
        "formation_energy_ev_natom" in sub_df.columns
    ), "Submission missing formation energy column"
    assert (
        "bandgap_energy_ev" in sub_df.columns
    ), "Submission missing bandgap energy column"

    # Check if we have predictions for the debug test set (should be 50 rows)
    # Note: If test.csv has fewer than 50 rows, it will be that number.
    # Based on metadata, test set has 240 rows, so debug should have 50.
    expected_rows = (
        50
        if len(pd.read_csv(Config.TEST_METADATA_PATH)) >= 50
        else len(pd.read_csv(Config.TEST_METADATA_PATH))
    )
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} predictions in debug mode, got {len(sub_df)}"

    print("Submission generation verified.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
