import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import calculate_log_mae, TypeSpecificStandardizer
from library.geometry import (
    get_radius_graph,
    get_triplets,
    compute_angles,
    SphericalBasisLayer,
    GaussianSmearing,
)
from library.data import ChampsDataset, get_collate_fn
from library.model import PhysicsAwareNet
from library.train import Trainer


def run_demonstration():
    print("=== Starting Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Set seeds for reproducibility
    Config.set_seed(42)

    # Define temporary working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_META_PATH = os.path.join(DEMO_DIR, "train_subset.csv")
    Config.VAL_META_PATH = os.path.join(DEMO_DIR, "val_subset.csv")

    # Override Cache paths in Config to point to demo dir
    # (Note: ChampsDataset uses Config.WORKING_DIR internally for cache prefix)
    Config.CACHE_NODES_PATH = os.path.join(DEMO_DIR, "cached_nodes.npy")
    Config.STATS_PATH = os.path.join(DEMO_DIR, "target_stats.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Reduce hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.MAX_EPOCHS = 1
    Config.HIDDEN_DIM = 32  # Smaller model for speed
    Config.NUM_LAYERS = 2

    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Verify Geometry Modules
    # ---------------------------------------------------------
    print("\n[2] Verifying Geometry Modules...")

    # Create dummy atoms (3 atoms in a line: 0,0,0 -> 1,0,0 -> 2,0,0)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32
    )

    # Test Radius Graph (Cutoff=2.0, so 0-1 connected, 1-2 connected, 0-2 not)
    edge_index, edge_dist = get_radius_graph(pos, cutoff=2.0)

    # Expect 4 edges (0->1, 1->0, 1->2, 2->1)
    assert edge_index.shape[1] == 4, f"Expected 4 edges, got {edge_index.shape[1]}"
    assert torch.allclose(
        edge_dist, torch.tensor([1.5, 1.5, 1.5, 1.5])
    ), "Distances incorrect"

    # Test Triplets
    triplets = get_triplets(edge_index, num_nodes=3)
    # 0->1->2 and 2->1->0 should be valid triplets.
    # 1->0 and 0->1 is backtracking, usually filtered or handled.
    # The get_triplets function filters k!=i.
    # Expected triplets: (0->1, 1->2) and (2->1, 1->0)
    assert triplets.shape[0] == 2, f"Expected 2 triplets, got {triplets.shape[0]}"

    # Test Angles
    angles = compute_angles(pos, edge_index, triplets)
    # Atoms are linear, angle should be pi (180 degrees) or close to it depending on vector direction
    # 0->1 vector is (1.5, 0, 0), 1->2 vector is (1.5, 0, 0).
    # compute_angles calculates angle for k->j->i.
    # v_ji = pos_i - pos_j, v_jk = pos_k - pos_j.
    # For 0->1->2: j=1, k=0, i=2. v_12 = (1.5,0,0), v_10 = (-1.5,0,0). Dot product negative. Angle pi.
    assert torch.allclose(
        angles, torch.tensor([np.pi, np.pi]), atol=1e-3
    ), f"Angles incorrect: {angles}"

    # Test Spherical Basis Layer
    sbf = SphericalBasisLayer(num_spherical=3, num_radial=4, cutoff=2.0)
    # Input: dist [T], angle [T]
    dist_input = torch.tensor([1.5, 1.5])
    angle_input = torch.tensor([np.pi, np.pi])
    feats = sbf(dist_input, angle_input)
    expected_dim = 3 * 4  # num_spherical * num_radial
    assert feats.shape == (
        2,
        expected_dim,
    ), f"SBF output shape mismatch. Got {feats.shape}"

    print("Geometry tests passed.")

    # ---------------------------------------------------------
    # 3. Verify Utils
    # ---------------------------------------------------------
    print("\n[3] Verifying Utils...")

    # Test Log MAE
    y_true = np.array([10.0, 1.0])
    y_pred = np.array([10.0, 1.0])  # Perfect prediction
    types = np.array([0, 1])
    score = calculate_log_mae(y_true, y_pred, types)
    # log(0 + 1e-9) is large negative, but let's check exact match logic
    assert score < -10.0, "Perfect prediction should have very low LogMAE"

    # Test Standardizer
    df_dummy = pd.DataFrame(
        {
            "type": ["1JHC", "1JHC", "2JHH"],
            "scalar_coupling_constant": [100.0, 110.0, 5.0],
        }
    )
    std_scaler = TypeSpecificStandardizer(device=torch.device("cpu"))
    std_scaler.fit(df_dummy, load_cached_data=False)

    # Check stats for 1JHC (Indices mapped in Config)
    idx_1jhc = Config.COUPLING_TYPE_MAP["1JHC"]
    mean_1jhc = std_scaler.means[idx_1jhc].item()
    assert abs(mean_1jhc - 105.0) < 1e-5, "Standardizer mean calculation failed"

    # Test Transform & Inverse
    vals = torch.tensor([105.0, 5.0])
    type_idxs = torch.tensor(
        [Config.COUPLING_TYPE_MAP["1JHC"], Config.COUPLING_TYPE_MAP["2JHH"]]
    )
    transformed = std_scaler.transform(vals, type_idxs)
    reconstructed = std_scaler.inverse_transform(transformed, type_idxs)
    assert torch.allclose(
        vals, reconstructed, atol=1e-5
    ), "Standardizer inverse transform failed"

    print("Utils tests passed.")

    # ---------------------------------------------------------
    # 4. Data Preparation (Subset)
    # ---------------------------------------------------------
    print("\n[4] Preparing Data Subset...")

    # Load original metadata to sample real molecules
    full_train_meta = pd.read_csv("./metadata/train_metadata.csv")

    # Pick 5 unique molecules for training, 2 for validation
    unique_mols = full_train_meta["molecule_name"].unique()
    train_mols = unique_mols[:5]
    val_mols = unique_mols[5:7]

    train_subset = full_train_meta[
        full_train_meta["molecule_name"].isin(train_mols)
    ].copy()
    val_subset = full_train_meta[full_train_meta["molecule_name"].isin(val_mols)].copy()

    print(
        f"Created subset: {len(train_subset)} train rows ({len(train_mols)} mols), "
        f"{len(val_subset)} val rows ({len(val_mols)} mols)"
    )

    # Save subsets
    train_subset.to_csv(Config.TRAIN_META_PATH, index=False)
    val_subset.to_csv(Config.VAL_META_PATH, index=False)

    # ---------------------------------------------------------
    # 5. Dataset & DataLoader
    # ---------------------------------------------------------
    print("\n[5] Initializing Dataset (Processing Subset)...")

    # Initialize Dataset (this triggers _process_data since cache doesn't exist in DEMO_DIR)
    # We suppress warnings from pandas/numpy during processing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_train = ChampsDataset(
            Config.TRAIN_META_PATH, mode="train", load_cached_data=False
        )

    assert (
        len(ds_train) == 5
    ), f"Dataset length mismatch. Expected 5, got {len(ds_train)}"

    # Check one item
    sample = ds_train[0]
    required_keys = ["x", "edge_index", "edge_attr", "triplets", "coupling_value"]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    print(
        f"Sample molecule has {sample['num_nodes']} nodes and {sample['coupling_value'].shape[0]} targets."
    )

    # Test Collate
    collate_fn = get_collate_fn()
    batch_list = [ds_train[0], ds_train[1]]
    batch = collate_fn(batch_list)

    assert (
        batch["batch"].shape[0] == sample["num_nodes"] + ds_train[1]["num_nodes"]
    ), "Batching nodes failed"
    print("Dataset and Collate tests passed.")

    # ---------------------------------------------------------
    # 6. Model Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Model Forward Pass...")

    device = Config.get_device()
    model = PhysicsAwareNet().to(device)

    # Move batch to device
    batch_x = batch["x"].to(device)
    edge_index = batch["edge_index"].to(device)
    edge_attr = batch["edge_attr"].to(device)
    triplets = batch["triplets"].to(device)
    triplet_attr = batch["triplet_attr"].to(device)
    coupling_index = batch["coupling_index"].to(device)
    coupling_type = batch["coupling_type"].to(device)

    # Forward
    pred_c, pred_s, pred_ch = model(
        x=batch_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        triplets=triplets,
        triplet_attr=triplet_attr,
        coupling_index=coupling_index,
        coupling_type=coupling_type,
    )

    # Check shapes
    num_couplings = batch["coupling_value"].shape[0]
    num_nodes = batch["x"].shape[0]

    assert pred_c.shape == (
        num_couplings,
    ), f"Coupling pred shape mismatch: {pred_c.shape}"
    assert pred_s.shape == (
        num_nodes,
        9,
    ), f"Shielding pred shape mismatch: {pred_s.shape}"
    assert pred_ch.shape == (
        num_nodes,
        1,
    ), f"Charge pred shape mismatch: {pred_ch.shape}"

    print("Model forward pass successful.")

    # ---------------------------------------------------------
    # 7. Integration: Trainer Loop
    # ---------------------------------------------------------
    print("\n[7] Running Trainer Integration Test...")

    trainer = Trainer()

    # Setup data (uses the Config paths we overrode)
    # We use load_cached_data=True because we just processed it in step 5
    # (ChampsDataset saves to cache automatically).
    trainer.setup_data(load_cached_data=True)

    # Setup model
    trainer.setup_model()

    # Run one epoch
    print("Running 1 training epoch...")
    loss, main_loss, aux_loss = trainer.train_epoch(1)

    print(f"Epoch 1 finished. Total Loss: {loss:.4f}")
    assert not np.isnan(loss), "Loss is NaN"

    # Run validation
    print("Running validation...")
    val_score = trainer.validate()
    print(f"Validation LogMAE: {val_score:.4f}")

    # Checkpoint saving test
    torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not saved"

    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    run_demonstration()
