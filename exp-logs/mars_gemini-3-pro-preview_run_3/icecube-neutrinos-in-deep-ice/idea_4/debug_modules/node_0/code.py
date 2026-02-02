import os
import sys
import torch
import numpy as np
import pandas as pd
import math

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model_lib
import library.train as train_lib


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup and Configuration
    # Set seed for reproducibility
    train_lib.set_seed(42)

    # Override config for speed and demo purposes
    config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 events for data loading demo
    config.BATCH_SIZE = 16  # Small batch size
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.EPOCHS = 2  # Minimal epochs for pipeline test

    print(
        f"Configuration set: Debug Size={config.DEBUG_SAMPLE_SIZE}, Batch Size={config.BATCH_SIZE}, Epochs={config.EPOCHS}"
    )

    # 2. Verify Utility Functions
    print("\n--- Verifying Utility Functions ---")

    # Test 1: Coordinate Conversion Round-trip
    # Vector along X-axis: Azimuth=0, Zenith=pi/2 -> x=1, y=0, z=0
    az_true, zen_true = 0.0, np.pi / 2
    x, y, z = utils.spherical_to_cartesian(az_true, zen_true)

    assert np.isclose(x, 1.0, atol=1e-6), f"Expected x=1.0, got {x}"
    assert np.isclose(y, 0.0, atol=1e-6), f"Expected y=0.0, got {y}"
    assert np.isclose(z, 0.0, atol=1e-6), f"Expected z=0.0, got {z}"

    az_rec, zen_rec = utils.cartesian_to_spherical(x, y, z)
    assert np.isclose(
        az_rec, az_true, atol=1e-6
    ), f"Expected Azimuth={az_true}, got {az_rec}"
    assert np.isclose(
        zen_rec, zen_true, atol=1e-6
    ), f"Expected Zenith={zen_true}, got {zen_rec}"

    print("Coordinate conversion verified.")

    # Test 2: Angular Distance Score
    # Distance between identical vectors should be 0
    y_true = np.array([[0.0, 1.0], [np.pi, 0.5]])  # [az, zen]
    y_pred_perfect = np.array([[0.0, 1.0], [np.pi, 0.5]])
    score_perfect = utils.angular_dist_score(y_true, y_pred_perfect)
    assert np.isclose(
        score_perfect, 0.0, atol=1e-6
    ), f"Expected score 0.0, got {score_perfect}"

    # Distance between opposite vectors along Z (Zenith 0 vs Zenith pi) should be pi
    y_up = np.array([[0.0, 0.0]])  # Up
    y_down = np.array([[0.0, np.pi]])  # Down
    score_opp = utils.angular_dist_score(y_up, y_down)
    assert np.isclose(
        score_opp, np.pi, atol=1e-6
    ), f"Expected score pi, got {score_opp}"

    print("Angular distance score verified.")

    # 3. Verify Data Loading
    print("\n--- Verifying Data Loading ---")

    # Initialize DataLoaders
    # This implicitly tests IceCubeGraphDataset, GroupedBatchSampler, and collate_fn
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=False)

    print(f"Train Loader Length: {len(train_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Structure (PyG Batch object)
    print(f"Batch keys: {batch.keys()}")

    # Check dimensions
    # x should be (Total_Nodes, NODE_FEAT_DIM)
    # NODE_FEAT_DIM is 5: [x, y, z, time, charge]
    assert (
        batch.x.shape[1] == config.NODE_FEAT_DIM
    ), f"Expected node feature dim {config.NODE_FEAT_DIM}, got {batch.x.shape[1]}"

    # global_features should be (Batch_Size, 12)
    # 3 eigenvalues + 9 eigenvector components
    expected_batch_size = batch.batch.max().item() + 1
    assert batch.global_features.shape == (
        expected_batch_size,
        12,
    ), f"Expected global features shape ({expected_batch_size}, 12), got {batch.global_features.shape}"

    # y should be (Batch_Size, 3) for Cartesian targets
    assert batch.y.shape == (
        expected_batch_size,
        3,
    ), f"Expected target shape ({expected_batch_size}, 3), got {batch.y.shape}"

    print("Data batch structure verified.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = model_lib.NeutrinoGNN().to(device)

    # Move batch to device
    batch = batch.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print(f"Model Output Shape: {output.shape}")

    # Assert output shape
    assert output.shape == (
        expected_batch_size,
        3,
    ), f"Expected output shape ({expected_batch_size}, 3), got {output.shape}"

    # Check Loss Function
    criterion = model_lib.CosineLoss()
    # Normalize output for loss calculation simulation (though loss handles unnormalized inputs)
    # We just pass raw output as per model_lib.CosineLoss logic
    loss = criterion(output, batch.y)
    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"

    print("Model forward pass and loss calculation verified.")

    # 5. Verify Full Training Pipeline
    print("\n--- Verifying Full Training Pipeline ---")

    # We use a slightly larger sample size for the pipeline run to ensure we have enough data for train/val splits
    # config.DEBUG_SAMPLE_SIZE was set to 100 earlier.
    # run_training allows overriding.

    # Ensure output directories are clean-ish (we don't delete, just overwrite)
    if os.path.exists(config.SUBMISSION_PATH):
        os.remove(config.SUBMISSION_PATH)

    # Run the pipeline
    # This function handles: Data Loading -> Training Loop -> Validation -> Inference -> Submission
    try:
        train_lib.run_training(epochs=2, debug_sample_size=250)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise e

    # 6. Verify Outputs
    print("\n--- Verifying Pipeline Outputs ---")

    # Check Model File
    if os.path.exists(config.MODEL_PATH):
        print(f"Model file found at: {config.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file not generated at {config.MODEL_PATH}")

    # Check Submission File
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"Submission file found at: {config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission shape: {df_sub.shape}")
        print(f"Submission columns: {list(df_sub.columns)}")

        # Basic content check
        assert "event_id" in df_sub.columns
        assert "azimuth" in df_sub.columns
        assert "zenith" in df_sub.columns
        assert len(df_sub) > 0
    else:
        raise FileNotFoundError(
            f"Submission file not generated at {config.SUBMISSION_PATH}"
        )

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    run_demo()
