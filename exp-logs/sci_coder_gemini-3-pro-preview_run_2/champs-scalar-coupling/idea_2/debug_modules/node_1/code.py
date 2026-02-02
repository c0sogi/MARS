import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import set_seed, GroupLogMAE
from library import geometry
from library.dataset import MoleculeDataset, collate_dmpnn
from library.model import DMPNN
from library.runner import Runner


def test_geometry_functions():
    """
    Validates the geometry processing functions (neighbors, triplets, angles)
    using a simple synthetic molecule (3 atoms forming a 90-degree angle).
    """
    print("\n=== Testing Geometry Functions ===")

    # Define 3 atoms: A(0,0,0), B(1,0,0), C(1,1,0)
    # Distance A-B = 1.0, B-C = 1.0, A-C = sqrt(2) ~ 1.414
    # Angle A-B-C should be 90 degrees (pi/2 radians)
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32
    )

    # 1. Test Neighbor Finding
    # Set cutoff to 1.1 to include A-B and B-C, but exclude A-C (1.414)
    cutoff = 1.1
    edge_index = geometry.get_neighbors(coords, cutoff=cutoff, max_neighbors=5)

    # Expect edges: (0->1), (1->0), (1->2), (2->1) => 4 edges
    print(f"Num edges found: {edge_index.shape[1]}")
    assert edge_index.shape == (2, 4), f"Expected shape (2, 4), got {edge_index.shape}"

    # Check distances
    dists = geometry.compute_distances(coords, edge_index)
    assert np.allclose(dists, 1.0), "All connected edges should have length 1.0"

    # 2. Test Triplet Finding
    # We expect triplets A->B->C (0->1->2) and C->B->A (2->1->0)
    # Note: A->B->A is excluded by logic (k != i)
    triplets, e1_idx, e2_idx = geometry.get_triplets(edge_index, num_atoms=3)

    print(f"Num triplets found: {triplets.shape[1]}")
    # Depending on edge ordering, we should find exactly 2 valid triplets for this chain
    assert triplets.shape[1] == 2, f"Expected 2 triplets, got {triplets.shape[1]}"

    # 3. Test Angle Computation
    angles = geometry.compute_angles(coords, triplets)
    expected_angle = np.pi / 2.0  # 90 degrees

    print(f"Computed angles: {angles}")
    assert np.allclose(
        angles, expected_angle
    ), f"Expected angles ~{expected_angle}, got {angles}"

    print("Geometry tests passed.")


def test_dataset_and_collate():
    """
    Validates the MoleculeDataset loading and the custom collate function.
    """
    print("\n=== Testing Dataset & Collate ===")

    # Use a tiny subset for testing
    debug_size = 10

    # Initialize dataset (uses Config paths setup in main)
    dataset = MoleculeDataset(
        Config.TRAIN_METADATA_PATH,
        mode="train",
        load_cached=False,  # Force re-computation to test logic
        debug_size=debug_size,
    )

    assert (
        len(dataset) == debug_size
    ), f"Dataset size mismatch: {len(dataset)} vs {debug_size}"

    # Test __getitem__
    sample = dataset[0]
    required_keys = [
        "atom_types",
        "coords",
        "edge_index",
        "edge_dists",
        "triplet_angles",
        "triplet_edge_index",
        "target",
        "type_idx",
        "target_edge_indices",
        "id",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    print(f"Sample 0 target: {sample['target']}")
    print(f"Sample 0 atoms: {sample['atom_types'].shape}")

    # Test Collate
    batch_size = 4
    batch_samples = [dataset[i] for i in range(batch_size)]
    batch = collate_dmpnn(batch_samples)

    # Verify Batch Shapes
    # targets should be (B,) or (B, 1) depending on implementation, here it's stacked
    assert batch["targets"].shape[0] == batch_size
    assert batch["ids"] is not None and len(batch["ids"]) == batch_size

    # Verify batch graph concatenation logic
    # Sum of atoms in individual samples should equal total atoms in batch
    total_atoms = sum(s["atom_types"].shape[0] for s in batch_samples)
    assert batch["atom_types"].shape[0] == total_atoms
    assert batch["batch_num_nodes"] == total_atoms

    print("Dataset and Collate tests passed.")
    return batch


def test_model_forward(batch):
    """
    Validates the DMPNN model forward pass and gradient propagation.
    """
    print("\n=== Testing Model Forward Pass ===")

    device = torch.device(Config.DEVICE)
    model = DMPNN().to(device)

    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Forward pass
    preds = model(batch)

    print(f"Prediction shape: {preds.shape}")
    assert preds.shape == (len(batch["targets"]), 1), "Output shape mismatch"

    # Backward pass check (ensure graph is connected and gradients flow)
    loss = torch.nn.L1Loss()(preds, batch["targets"].unsqueeze(-1).to(device))
    loss.backward()

    # Check if gradients exist for a key parameter
    param = list(model.embedding.atom_embedding.parameters())[0]
    assert param.grad is not None, "Gradients not flowing to embedding layer"

    print("Model forward/backward pass passed.")


def test_runner_execution():
    """
    Runs the full training and prediction pipeline using the Runner class.
    """
    print("\n=== Testing Full Runner Execution ===")

    # Initialize Runner with debug=True
    # This triggers the internal logic to load datasets with Config.DEBUG_SAMPLE_SIZE
    runner = Runner(debug=True, load_cached_data=True)

    # 1. Train
    print("Starting Runner.train()...")
    runner.train()

    # Check if model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"

    # 2. Predict
    print("Starting Runner.predict()...")
    runner.predict()

    # Check if submission was generated
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Submission file not found after prediction"

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    assert len(df_sub) > 0

    print("Runner execution passed.")


def test_metric_logic():
    """
    Validates the GroupLogMAE metric calculation.
    """
    print("\n=== Testing Metric Logic ===")
    metric = GroupLogMAE()

    # Case: 2 types.
    # Type 0: Error = 10 -> Log(10) = 2.302
    # Type 1: Error = 1  -> Log(1) = 0.0
    # Avg Log MAE = (2.302 + 0.0) / 2 = 1.151

    preds = torch.tensor([10.0, 1.0])
    targets = torch.tensor([0.0, 0.0])
    types = torch.tensor([0, 1])

    metric.update(preds, targets, types)
    score, type_metrics = metric.compute()

    print(f"Computed Score: {score:.4f}")
    assert 1.15 < score < 1.16, f"Metric calculation incorrect. Got {score}"
    assert len(type_metrics) == 2

    print("Metric logic passed.")


if __name__ == "__main__":
    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    # We modify the Config class attributes directly to ensure the demo runs fast.
    # These changes propagate to all modules that import Config.

    # Create a specific working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR

    # Set Paths
    Config.CACHE_TRAIN_DATA = os.path.join(DEMO_DIR, "cached_train_v2.npz")
    Config.CACHE_VAL_DATA = os.path.join(DEMO_DIR, "cached_val_v2.npz")
    Config.CACHE_TEST_DATA = os.path.join(DEMO_DIR, "cached_test_v2.npz")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pt")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Set Hyperparameters for Speed
    Config.DEBUG_SAMPLE_SIZE = 200  # Small subset
    Config.EPOCHS = 2  # Minimal epochs
    Config.WARMUP_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Re-run setup to ensure directories exist
    Config.setup()
    set_seed(Config.SEED)

    print(f"Configured for demo run in {Config.WORKING_DIR}")

    # ==========================================
    # 2. Execute Tests
    # ==========================================

    try:
        # Test 1: Geometry
        test_geometry_functions()

        # Test 2: Dataset & Collate
        batch = test_dataset_and_collate()

        # Test 3: Model
        test_model_forward(batch)

        # Test 4: Metric
        test_metric_logic()

        # Test 5: Full Runner (Train & Predict)
        test_runner_execution()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nRUNTIME ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
