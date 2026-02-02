import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# Import provided library modules
from library.config import Config
from library.utils import (
    azimuth_zenith_to_vector,
    vector_to_azimuth_zenith,
    angular_dist_score,
)
from library.data import IceCubeGraphDataset
from library.model import IceCubeDGCN
from library.engine import Engine


def main():
    print("=== Starting IceCube Pipeline Demo ===")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Redirect output to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Reduce data complexity
    Config.MAX_PULSES = 64  # Smaller graphs
    Config.K_NEIGHBORS = 6  # Fewer neighbors

    # Reduce model complexity
    Config.HIDDEN_DIM = 32
    Config.NUM_LAYERS = 2
    Config.DROPOUT = 0.0

    # Reduce training duration
    Config.BATCH_SIZE = 16
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Re-initialize to ensure directories exist
    Config.initialize()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[2] Verifying utility functions...")

    # Test 1: Coordinate Conversion
    # Vector along X-axis: (1, 0, 0) -> Azimuth 0, Zenith pi/2
    vec_x = torch.tensor([[1.0, 0.0, 0.0]])
    az, zen = vector_to_azimuth_zenith(vec_x)

    assert torch.isclose(
        az, torch.tensor([0.0]), atol=1e-4
    ).all(), f"Azimuth mismatch: {az}"
    assert torch.isclose(
        zen, torch.tensor([np.pi / 2]), atol=1e-4
    ).all(), f"Zenith mismatch: {zen}"

    # Round trip
    vec_recon = azimuth_zenith_to_vector(az, zen)
    assert torch.allclose(vec_x, vec_recon, atol=1e-4), "Round trip conversion failed"

    # Test 2: Angular Distance Score
    # Distance between (1,0,0) and (0,1,0) (90 degrees / pi/2 radians)
    y_true = np.array([[0.0, np.pi / 2]])  # Az=0, Zen=90 -> X axis
    y_pred = np.array([[np.pi / 2, np.pi / 2]])  # Az=90, Zen=90 -> Y axis

    score = angular_dist_score(y_true, y_pred)
    expected_score = np.pi / 2
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Score mismatch: {score} vs {expected_score}"

    print("Utility functions verified successfully.")

    # ==========================================
    # 3. Data Pipeline Demo
    # ==========================================
    print("\n[3] Setting up Data Pipeline...")

    # Load metadata to find a valid batch ID
    train_meta = pd.read_parquet(Config.TRAIN_META_PATH)
    sample_batch_id = train_meta["batch_id"].iloc[0]
    print(f"Using Batch ID {sample_batch_id} for demonstration.")

    # Initialize Dataset
    # This will trigger processing and caching of the selected batch
    dataset = IceCubeGraphDataset(mode="train", batch_ids=[sample_batch_id])

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)

    # Fetch a small subset of data to use for training/val
    # We don't want to iterate the whole dataset in this demo
    print("Fetching a subset of data (2 batches)...")
    demo_batches = []
    for i, batch in enumerate(loader):
        demo_batches.append(batch)
        if i >= 1:  # Keep only 2 batches
            break

    assert len(demo_batches) > 0, "DataLoader yielded no data."

    # Verify Data Structure
    sample_data = demo_batches[0]
    print(f"Sample Batch Structure: {sample_data}")

    # Check Feature Dimensions: [num_nodes, INPUT_DIM]
    assert (
        sample_data.x.shape[1] == Config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {sample_data.x.shape[1]}"

    # Check Target Dimensions: [batch_size, 3] (Unit vectors)
    assert sample_data.y.shape == (
        sample_data.num_graphs,
        3,
    ), f"Target shape mismatch. Expected ({sample_data.num_graphs}, 3), got {sample_data.y.shape}"

    print("Data pipeline verified.")

    # ==========================================
    # 4. Model & Engine Demo
    # ==========================================
    print("\n[4] Initializing Model and Engine...")

    model = IceCubeDGCN().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    engine = Engine(model=model, device=Config.DEVICE, optimizer=optimizer)

    # Run Training Loop
    # We pass the list of batches directly since it's iterable
    save_path = os.path.join(Config.WORKING_DIR, "model_checkpoints", "best_model.pth")

    print("Running Engine.fit() on demo subset...")
    engine.fit(
        train_loader=demo_batches,
        val_loader=demo_batches,  # Use same for demo
        epochs=Config.NUM_EPOCHS,
        patience=1,
        save_path=save_path,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print("Training loop completed successfully.")

    # ==========================================
    # 5. Inference Demo
    # ==========================================
    print("\n[5] Running Inference Demo...")

    submission_path = Config.SUBMISSION_PATH

    # Run prediction on the same demo batches
    engine.predict(demo_batches, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    expected_cols = ["event_id", "azimuth", "zenith"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch: {df_sub.columns}"
    assert len(df_sub) == sum(
        b.num_graphs for b in demo_batches
    ), "Submission row count mismatch."

    print("Inference verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
