import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys

# Import from the provided library
from library.config import Config
from library.utils import set_seed, TargetScaler
from library.data import get_dataloaders, CrystalGraphDataset
from library.model import SS_CGCNN
from library.train import run_training


def setup_demo_environment():
    """
    Sets up a temporary directory with a subset of metadata to run a fast demo.
    Modifies the Config class in-place to point to these demo files.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_run"
    demo_metadata_dir = os.path.join(demo_dir, "metadata")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_submission_dir = os.path.join(demo_dir, "submission")

    os.makedirs(demo_metadata_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create subsets (e.g., 20 samples each) for speed
    subset_size = 20
    demo_train_df = train_df.head(subset_size)
    demo_val_df = val_df.head(subset_size)
    demo_test_df = test_df.head(subset_size)

    # Save demo metadata
    demo_train_path = os.path.join(demo_metadata_dir, "train_metadata.csv")
    demo_val_path = os.path.join(demo_metadata_dir, "val_metadata.csv")
    demo_test_path = os.path.join(demo_metadata_dir, "test_metadata.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    # --- Monkey-patch Config ---
    # We modify the Config class attributes directly so that imported modules use the new paths
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_submission_dir

    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    Config.TRAIN_GRAPHS_CACHE = os.path.join(demo_cache_dir, "train_graphs.npz")
    Config.VAL_GRAPHS_CACHE = os.path.join(demo_cache_dir, "val_graphs.npz")
    Config.TEST_GRAPHS_CACHE = os.path.join(demo_cache_dir, "test_graphs.npz")
    Config.TARGET_SCALER_CACHE = os.path.join(demo_cache_dir, "target_scaler.npz")

    Config.MODEL_CHECKPOINT_PATH = os.path.join(
        demo_dir, "checkpoints", "best_model.pth"
    )
    os.makedirs(os.path.dirname(Config.MODEL_CHECKPOINT_PATH), exist_ok=True)

    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    # Reduce training parameters for demo speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 2

    print("Demo environment configured.")


def test_target_scaler():
    """
    Verifies the logic of the TargetScaler class.
    """
    print("\nTesting TargetScaler...")
    scaler = TargetScaler()

    # Create dummy data: 10 samples, 2 targets
    data = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        dtype=np.float32,
    )

    # Fit scaler
    scaler.fit(data)

    # Check mean and std
    expected_mean = np.array([3.0, 30.0])
    expected_std = np.std(data, axis=0)

    assert np.allclose(
        scaler.mean, expected_mean
    ), f"Scaler mean mismatch: {scaler.mean} vs {expected_mean}"

    # Transform
    transformed = scaler.transform(data)
    assert np.allclose(
        np.mean(transformed, axis=0), 0.0, atol=1e-6
    ), "Transformed mean should be 0"
    assert np.allclose(
        np.std(transformed, axis=0), 1.0, atol=1e-6
    ), "Transformed std should be 1"

    # Inverse Transform
    reconstructed = scaler.inverse_transform(transformed)
    assert np.allclose(
        data, reconstructed, atol=1e-5
    ), "Inverse transform failed to reconstruct data"

    # Test with Torch Tensors
    data_tensor = torch.tensor(data)
    transformed_tensor = scaler.transform(data_tensor)
    assert isinstance(transformed_tensor, torch.Tensor)
    reconstructed_tensor = scaler.inverse_transform(transformed_tensor)
    assert torch.allclose(data_tensor, reconstructed_tensor, atol=1e-5)

    print("TargetScaler tests passed.")


def test_model_architecture():
    """
    Instantiates the model and runs a dummy forward pass to verify architecture.
    """
    print("\nTesting SS_CGCNN Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SS_CGCNN(config=Config).to(device)

    # Create a dummy batch
    # 2 graphs in the batch
    # Graph 1: 3 nodes, 2 edges
    # Graph 2: 2 nodes, 1 edge
    x = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long).to(device)  # Atomic numbers
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    ).to(device)
    edge_attr = (
        torch.tensor([1.5, 1.5, 2.0, 2.0, 1.8, 1.8], dtype=torch.float)
        .unsqueeze(1)
        .to(device)
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long).to(device)

    # Mock Data object
    from torch_geometric.data import Data, Batch

    data1 = Data(
        x=x[:3],
        edge_index=edge_index[:, :4],
        edge_attr=edge_attr[:4],
        y=torch.zeros(1, 2),
    )
    data2 = Data(
        x=x[3:],
        edge_index=edge_index[:, 4:] - 3,
        edge_attr=edge_attr[4:],
        y=torch.zeros(1, 2),
    )
    batch_data = Batch.from_data_list([data1, data2]).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_data)

    # Check output shape: (batch_size, num_targets) -> (2, 2)
    assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
    print("Model forward pass successful. Output shape verified.")


def run_demo_pipeline():
    """
    Runs the full training pipeline using the demo configuration.
    """
    print("\nRunning Demo Training Pipeline...")

    # Ensure seed is set
    set_seed(Config.SEED)

    # 1. Data Loading (this will trigger processing since cache doesn't exist in demo dir)
    # We set load_cached_data=False to force processing logic verification
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Verify data content
    sample_batch = next(iter(train_loader))
    assert sample_batch.x.dim() == 1, "Node features should be 1D (atomic numbers)"
    assert sample_batch.y.shape[1] == 2, "Target should have 2 columns"

    # 2. Run Training
    # run_training handles model init, loop, saving, and submission generation
    best_loss = run_training(load_cached_data=True, num_epochs=Config.NUM_EPOCHS)

    print(f"Demo training finished. Best Val Loss: {best_loss:.4f}")

    # 3. Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated with {len(sub_df)} rows.")
        assert (
            len(sub_df) == 20
        ), f"Expected 20 rows in demo submission, got {len(sub_df)}"
        assert list(sub_df.columns) == [
            "id",
            "formation_energy_ev_natom",
            "bandgap_energy_ev",
        ]
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Verify Utils
    test_target_scaler()

    # 3. Verify Model
    test_model_architecture()

    # 4. Run Pipeline
    run_demo_pipeline()

    print("\nAll demo steps completed successfully.")
