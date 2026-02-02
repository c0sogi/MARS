import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import GroupStandardizer, calculate_log_mae
from library.layers import BesselBasisLayer, SphericalBasisLayer, InteractionBlock
from library.model import DirectionalMPNN
from library.train import run_training, set_seed
from library.predict import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_data(working_dir, num_molecules=50):
    """
    Creates a small subset of the data in the working directory for fast demonstration.
    """
    print(f"Creating data subset with {num_molecules} molecules...")

    # 1. Load Structures and pick top N molecules
    df_struct = pd.read_csv(Config.STRUCTURES_CSV)
    unique_mols = df_struct["molecule_name"].unique()[:num_molecules]
    subset_struct = df_struct[df_struct["molecule_name"].isin(unique_mols)].copy()

    subset_struct_path = os.path.join(working_dir, "structures.csv")
    subset_struct.to_csv(subset_struct_path, index=False)

    # 2. Filter Metadata
    for meta_type, orig_path in [
        ("train", Config.TRAIN_META_PATH),
        ("val", Config.VAL_META_PATH),
        ("test", Config.TEST_META_PATH),
    ]:
        df_meta = pd.read_csv(orig_path)
        subset_meta = df_meta[df_meta["molecule_name"].isin(unique_mols)].copy()

        # Save to working dir
        save_path = os.path.join(working_dir, f"{meta_type}_metadata.csv")
        subset_meta.to_csv(save_path, index=False)

    return subset_struct_path


def test_utils():
    print("\n--- Testing Utils ---")

    # Test GroupStandardizer
    print("Testing GroupStandardizer...")
    types = ["1JHC", "1JHC", "2JHH", "2JHH"]
    values = [10.0, 20.0, 5.0, 15.0]

    df = pd.DataFrame({"type": types, "scalar_coupling_constant": values})

    gs = GroupStandardizer()
    # Mock the cache file path to avoid overwriting real cache
    gs.stats_file = os.path.join(Config.WORKING_DIR, "test_stats.npy")
    if os.path.exists(gs.stats_file):
        os.remove(gs.stats_file)

    gs.fit(df, load_cached_data=False)

    # Check means
    assert gs.means["1JHC"] == 15.0, "Mean calculation failed for 1JHC"
    assert gs.means["2JHH"] == 10.0, "Mean calculation failed for 2JHH"

    # Test Transform
    transformed = gs.transform(values, types)
    # 1JHC: (10-15)/std, (20-15)/std. std of [10, 20] is sqrt(50) ~= 7.07
    # 2JHH: (5-10)/std, (15-10)/std. std of [5, 15] is sqrt(50) ~= 7.07
    assert np.allclose(
        transformed[0], (10 - 15) / np.std([10, 20], ddof=1)
    ), "Transform failed"

    # Test Inverse Transform
    inversed = gs.inverse_transform(transformed, types)
    assert np.allclose(inversed, values), "Inverse transform failed"

    # Test Metric
    print("Testing calculate_log_mae...")
    y_true = np.array([10.0, 100.0])
    y_pred = np.array([11.0, 110.0])  # MAE = 1, 10
    t_types = np.array(["A", "B"])
    # LogMAE = mean(log(1), log(10)) = mean(0, 2.302) = 1.151
    metric = calculate_log_mae(y_true, y_pred, t_types)
    assert metric > 0, "Metric calculation failed"
    print("Utils verified.")


def test_layers():
    print("\n--- Testing Layers ---")

    # Configuration
    num_radial = 8
    num_spherical = 4
    hidden_channels = 16
    cutoff = 5.0

    # 1. Bessel Basis
    print("Testing BesselBasisLayer...")
    rbf = BesselBasisLayer(num_radial, cutoff)
    dist = torch.rand(10) * cutoff
    out_rbf = rbf(dist)
    assert out_rbf.shape == (10, num_radial), f"RBF shape mismatch: {out_rbf.shape}"

    # 2. Spherical Basis
    print("Testing SphericalBasisLayer...")
    sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff)
    angle = torch.rand(20) * 3.14
    # idx_kj maps triplets to edges. Let's assume we have 10 edges and 20 triplets
    idx_kj = torch.randint(0, 10, (20,))
    # We need dist for the edges referenced by idx_kj
    dist_edges = torch.rand(10) * cutoff

    out_sbf = sbf(dist_edges, angle, idx_kj)
    expected_dim = num_radial * num_spherical
    assert out_sbf.shape == (20, expected_dim), f"SBF shape mismatch: {out_sbf.shape}"

    # 3. Interaction Block
    print("Testing InteractionBlock...")
    block = InteractionBlock(
        hidden_channels, num_radial, num_spherical, num_bilinear=hidden_channels
    )

    # Dummy inputs
    x = torch.randn(10, hidden_channels)  # Edge embeddings
    rbf_emb = torch.randn(10, num_radial)
    sbf_emb = torch.randn(20, expected_dim)
    idx_ji = torch.randint(0, 10, (20,))  # Target edges for triplets

    out_x = block(x, rbf_emb, sbf_emb, idx_kj, idx_ji)
    assert out_x.shape == (
        10,
        hidden_channels,
    ), f"Interaction output mismatch: {out_x.shape}"
    print("Layers verified.")


def test_model_forward():
    print("\n--- Testing Model Forward Pass ---")

    # Setup small model
    model = DirectionalMPNN(
        hidden_channels=16, num_layers=2, num_radial=8, num_spherical=4, out_emb_dim=8
    )

    # Create dummy graph
    # 3 Atoms: 0, 1, 2
    # Edges: 0->1, 1->0, 1->2, 2->1
    # Triplets: 0->1->2, 2->1->0

    z = torch.tensor([1, 6, 1], dtype=torch.long)  # H, C, H
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float
    )

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # (2, E)

    # Triplets (k->j->i)
    # 0->1 (edge 0) and 1->2 (edge 2) -> Triplet 0->1->2
    # 2->1 (edge 3) and 1->0 (edge 1) -> Triplet 2->1->0
    idx_kj = torch.tensor([0, 3], dtype=torch.long)
    idx_ji = torch.tensor([2, 1], dtype=torch.long)

    # Target: Pair (0, 2)
    target_node_0 = torch.tensor([0], dtype=torch.long)
    target_node_1 = torch.tensor([2], dtype=torch.long)
    target_type = torch.tensor([0], dtype=torch.long)  # Type 0

    # Target edge indices (direct edges usually don't exist for long range, but model handles it)
    # Here 0->2 doesn't exist in graph.
    target_edge_index_uv = torch.tensor([-1], dtype=torch.long)
    target_edge_index_vu = torch.tensor([-1], dtype=torch.long)

    out = model(
        z=z,
        pos=pos,
        edge_index=edge_index,
        idx_kj=idx_kj,
        idx_ji=idx_ji,
        target_node_0=target_node_0,
        target_node_1=target_node_1,
        target_type=target_type,
        target_edge_index_uv=target_edge_index_uv,
        target_edge_index_vu=target_edge_index_vu,
    )

    assert out.shape == (1, 1), f"Model output shape mismatch: {out.shape}"
    print("Model forward pass verified.")


def run_pipeline_demo():
    print("\n--- Running Full Pipeline Demo ---")

    # 1. Setup paths and config
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config to use the demo directory and subset data
    Config.WORKING_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.PROCESSED_DATA_CACHE = os.path.join(demo_dir, "processed_graphs.pt")
    Config.STATS_CACHE = os.path.join(demo_dir, "target_stats.npy")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Create subset
    subset_struct_path = create_subset_data(demo_dir, num_molecules=50)

    Config.STRUCTURES_CSV = subset_struct_path
    Config.TRAIN_META_PATH = os.path.join(demo_dir, "train_metadata.csv")
    Config.VAL_META_PATH = os.path.join(demo_dir, "val_metadata.csv")
    Config.TEST_META_PATH = os.path.join(demo_dir, "test_metadata.csv")

    # Speed up training config
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.HIDDEN_CHANNELS = 16
    Config.NUM_LAYERS = 2
    Config.NUM_RBF = 8
    Config.NUM_SBF = 4
    Config.DEBUG_SAMPLE_SIZE = None  # We already subsetted the files
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 2. Run Training
    print("Starting training...")
    # Force reload cache since we changed data
    best_metric = run_training(load_cached_data=False)
    print(f"Training complete. Best Metric: {best_metric}")

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."

    # 3. Run Prediction
    print("Starting prediction...")
    generate_submission(load_cached_data=True, batch_size=4, device=Config.DEVICE)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")
    assert len(df_sub) > 0, "Submission file is empty."
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission columns missing."

    print("Pipeline demo verified.")


if __name__ == "__main__":
    # Set global seed
    set_seed(Config.SEED)

    # Run verifications
    test_utils()
    test_layers()
    test_model_forward()

    # Run full integration
    run_pipeline_demo()

    print("\nAll tests passed successfully.")
