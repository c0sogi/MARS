import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_log_mae, Standardizer
from library import features
from library.preprocessing import process_dataset
from library.dataset import FlattenedMoleculeDataset, collate_batch
from library.model import MoleculeModel
from library.train import train_model


def run_demo():
    # ==========================================
    # 0. Setup & Configuration Override
    # ==========================================
    print(">>> Setting up configuration for demo...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set fixed seeds
    set_seed(42)

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset
    Config.WORK_DIR = "./working/demo_run"
    Config.PROCESSED_DATA_DIR = os.path.join(Config.WORK_DIR, "processed")

    # Update cache paths to point to the demo directory
    Config.CACHE_PATHS = {
        "train": os.path.join(Config.PROCESSED_DATA_DIR, "train_data"),
        "val": os.path.join(Config.PROCESSED_DATA_DIR, "val_data"),
        "test": os.path.join(Config.PROCESSED_DATA_DIR, "test_data"),
        "stats": os.path.join(Config.PROCESSED_DATA_DIR, "stats.npy"),
    }

    # Reduce Model Complexity for Speed
    Config.HIDDEN_DIM = 32
    Config.NUM_INTERACTIONS = 2
    Config.NUM_RBF = 16
    Config.NUM_ANGLE_RBF = 8
    Config.NUM_HEADS = 4

    # Reduce Training params
    Config.BATCH_SIZE = 8
    Config.MAX_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.DEVICE = "cpu"  # Use CPU for simple demo stability, or "cuda" if available
    if torch.cuda.is_available():
        Config.DEVICE = "cuda"

    # Clean up previous demo run
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)

    print(f"Configured. Work Dir: {Config.WORK_DIR}, Device: {Config.DEVICE}")

    # ==========================================
    # 1. Feature Logic Verification
    # ==========================================
    print("\n>>> Verifying Feature Logic (Geometry)...")

    # Create a synthetic right-angle triangle: A(0,0,0) -> B(1,0,0) -> C(1,1,0)
    # Distance AB = 1.0, BC = 1.0, AC = sqrt(2) approx 1.414
    # Angle at B (A-B-C) should be 90 degrees (pi/2 radians)

    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1 (Central atom for angle)
            [1.0, 1.0, 0.0],  # 2
        ],
        dtype=torch.float32,
    )

    # Define edges explicitly: 0->1 and 1->2
    # Edge 0: 0->1 (k->j)
    # Edge 1: 1->2 (j->i)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    # Compute Distances
    dist, vec = features.compute_dist(pos, edge_index)

    # Verify Distances
    assert torch.isclose(
        dist[0], torch.tensor(1.0)
    ), f"Expected dist 1.0, got {dist[0]}"
    assert torch.isclose(
        dist[1], torch.tensor(1.0)
    ), f"Expected dist 1.0, got {dist[1]}"

    # Define Triplet: Edge 0 (0->1) feeds into Edge 1 (1->2) at node 1
    # triplet_indices: [incoming_edge_idx, outgoing_edge_idx]
    triplet_indices = torch.tensor([[0], [1]], dtype=torch.long)

    # Compute Angles
    angles = features.compute_angles(pos, edge_index, triplet_indices)

    # Verify Angle (pi/2 = 1.57079...)
    expected_angle = np.pi / 2.0
    assert torch.isclose(
        angles[0], torch.tensor(expected_angle), atol=1e-4
    ), f"Expected angle {expected_angle}, got {angles[0]}"

    print("Feature logic verified successfully.")

    # ==========================================
    # 2. Preprocessing Verification
    # ==========================================
    print("\n>>> Verifying Preprocessing...")

    # Process train split (this will create cache files)
    data_dict = process_dataset("train", load_cached_data=False)

    # Check keys
    required_keys = ["node_z", "edge_index", "target_values", "target_indices"]
    for k in required_keys:
        assert k in data_dict, f"Missing key {k} in processed data"

    # Check shapes
    num_nodes = data_dict["node_z"].shape[0]
    num_edges = data_dict["edge_index"].shape[1]
    num_targets = data_dict["target_values"].shape[0]

    print(f"Processed {num_nodes} nodes, {num_edges} edges, {num_targets} targets.")
    assert num_nodes > 0
    assert num_edges > 0
    assert num_targets > 0

    # Verify normalization/standardizer stats generation
    assert os.path.exists(Config.CACHE_PATHS["stats"]), "Stats file not generated"

    # ==========================================
    # 3. Dataset & DataLoader Verification
    # ==========================================
    print("\n>>> Verifying Dataset & DataLoader...")

    dataset = FlattenedMoleculeDataset(split="train", load_cached=True)
    assert len(dataset) > 0, "Dataset is empty"

    # Get one item
    item = dataset[0]
    assert "node_z" in item
    assert "edge_index" in item
    assert item["edge_index"].max() < item["num_nodes"], "Edge index out of bounds"

    # Test Collation
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch
    )
    batch = next(iter(loader))

    assert batch["batch_size"] == Config.BATCH_SIZE or batch["batch_size"] == len(
        dataset
    )
    assert "node_batch" in batch
    assert batch["node_batch"].max() < batch["batch_size"]

    print(
        f"Batch loaded. Nodes: {batch['node_z'].shape}, Edges: {batch['edge_index'].shape}"
    )

    # ==========================================
    # 4. Model Verification
    # ==========================================
    print("\n>>> Verifying Model...")

    model = MoleculeModel().to(Config.DEVICE)

    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(Config.DEVICE)

    # Forward pass
    preds = model(batch)

    # Check outputs
    assert "scalar_coupling" in preds
    assert preds["scalar_coupling"].shape == (batch["target_values"].shape[0], 1)

    # Check aux outputs
    assert "charges" in preds
    assert "shielding" in preds

    # Check backward pass capability
    loss = preds["scalar_coupling"].sum()
    loss.backward()
    print("Model forward and backward pass successful.")

    # ==========================================
    # 5. Utils Verification
    # ==========================================
    print("\n>>> Verifying Utils...")

    # Test Standardizer
    std = Standardizer()
    # Mock data: type 'A' has mean 10, std 2
    df_mock = pd.DataFrame(
        {"type": ["A", "A", "A"], "scalar_coupling_constant": [8.0, 10.0, 12.0]}
    )
    std.fit(df_mock)

    vals = np.array([14.0])
    types = np.array(["A"])
    trans = std.transform(vals, types)
    # (14 - 10) / 2 = 2.0
    assert np.isclose(trans[0], 2.0), f"Standardizer failed. Got {trans[0]}"

    inv = std.inverse_transform(trans, types)
    assert np.isclose(inv[0], 14.0), f"Inverse transform failed. Got {inv[0]}"

    # Test Metric
    y_true = np.array([10.0, 100.0])
    y_pred = np.array([10.0, 100.0])  # Perfect prediction
    types_arr = np.array(["1JHC", "2JHC"])

    score = calculate_log_mae(y_true, y_pred, types_arr)
    # log(0 + 1e-9) -> -9.0 approx
    assert score < -8.0, f"Metric should be very low for perfect pred, got {score}"

    print("Utils verified.")

    # ==========================================
    # 6. Full Training Pipeline Verification
    # ==========================================
    print("\n>>> Verifying Full Training Pipeline...")

    # This runs the training loop, validation, and submission generation
    # It uses the Config overrides we set earlier (1 epoch, small batch)
    train_model()

    # Check artifacts
    assert os.path.exists(
        os.path.join(Config.WORK_DIR, "best_model.pth")
    ), "Model checkpoint not saved"
    assert os.path.exists(
        os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    ), "Submission file not created"

    # Check submission content
    sub_df = pd.read_csv(os.path.join(Config.SUBMISSION_DIR, "submission.csv"))
    assert "id" in sub_df.columns
    assert "scalar_coupling_constant" in sub_df.columns
    assert len(sub_df) > 0

    print("Full training pipeline completed successfully.")
    print("\n>>> DEMO COMPLETED SUCCESSFULLY <<<")


if __name__ == "__main__":
    run_demo()
