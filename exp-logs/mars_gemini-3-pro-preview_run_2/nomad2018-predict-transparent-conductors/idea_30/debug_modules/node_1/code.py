import os
import pandas as pd
import torch
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import TargetScaler, set_seed
from library.data import process_structure, get_dataloaders
from library.model import IS_RA_CGN
from library.train import run_training


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo run by modifying the Config
    and creating a subset of the metadata to ensure quick execution.
    """
    print("--- Setting up Demo Environment ---")

    # 1. Modify Config class attributes for the demo
    # Since the library modules use an instance of Config that relies on class attributes,
    # modifying the class attributes here propagates to the other modules.
    Config.num_epochs = 2
    Config.batch_size = 4
    Config.working_dir = "./working/demo_run"
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.checkpoint_dir = os.path.join(Config.working_dir, "checkpoints")
    Config.submission_dir = os.path.join(Config.working_dir, "submission")

    # Ensure these directories exist
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    # 2. Create a small subset of metadata
    demo_meta_dir = "./working/demo_metadata"
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Load original metadata
    # We assume the metadata files exist as per the problem description
    train_full = pd.read_csv("./metadata/train_metadata.csv")
    val_full = pd.read_csv("./metadata/val_metadata.csv")
    test_full = pd.read_csv("./metadata/test_metadata.csv")

    # Take a small sample (e.g., 20 training, 10 val, 10 test)
    train_demo = train_full.head(20)
    val_demo = val_full.head(10)
    test_demo = test_full.head(10)

    # Save demo metadata
    Config.train_metadata_path = os.path.join(demo_meta_dir, "train_metadata.csv")
    Config.val_metadata_path = os.path.join(demo_meta_dir, "val_metadata.csv")
    Config.test_metadata_path = os.path.join(demo_meta_dir, "test_metadata.csv")

    train_demo.to_csv(Config.train_metadata_path, index=False)
    val_demo.to_csv(Config.val_metadata_path, index=False)
    test_demo.to_csv(Config.test_metadata_path, index=False)

    print(f"Demo metadata created at {demo_meta_dir}")
    print(f"Config updated: Epochs={Config.num_epochs}, Batch Size={Config.batch_size}")


def test_target_scaler():
    """
    Verifies the logic of the TargetScaler class.
    """
    print("\n--- Testing TargetScaler ---")
    scaler = TargetScaler()

    # Create dummy data: 3 samples, 2 targets
    # Target 1: 1, 2, 3 -> Mean 2, Std 1
    # Target 2: 10, 20, 30 -> Mean 20, Std 10
    y = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float32)

    scaler.fit(y)

    expected_mean = torch.tensor([2.0, 20.0])
    expected_std = torch.tensor([1.0, 10.0])

    if not torch.allclose(scaler.mean, expected_mean):
        raise AssertionError(
            f"Scaler mean mismatch. Expected {expected_mean}, got {scaler.mean}"
        )
    if not torch.allclose(scaler.std, expected_std):
        raise AssertionError(
            f"Scaler std mismatch. Expected {expected_std}, got {scaler.std}"
        )

    # Test Transform
    y_scaled = scaler.transform(y)
    expected_scaled = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    if not torch.allclose(y_scaled, expected_scaled, atol=1e-5):
        raise AssertionError(f"Scaler transform mismatch. Got \n{y_scaled}")

    # Test Inverse Transform
    y_inv = scaler.inverse_transform(y_scaled)
    if not torch.allclose(y_inv, y, atol=1e-5):
        raise AssertionError(
            "Scaler inverse transform failed to recover original data."
        )

    print("TargetScaler logic verified.")


def test_data_processing():
    """
    Verifies that a structure file can be processed into a graph.
    """
    print("\n--- Testing Data Processing ---")
    # Read the first entry from our demo train metadata
    df = pd.read_csv(Config.train_metadata_path)
    row = df.iloc[0]
    file_path = row["file_path"]
    targets = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]

    print(f"Processing file: {file_path}")
    data = process_structure(file_path, targets)

    if data is None:
        raise RuntimeError("process_structure returned None.")

    # Check Data object attributes
    print(f"Graph attributes: {data}")
    if not hasattr(data, "x") or data.x is None:
        raise AssertionError("Graph missing node features 'x'")
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise AssertionError("Graph missing 'edge_index'")
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        raise AssertionError("Graph missing 'edge_attr'")
    if not hasattr(data, "y") or data.y is None:
        raise AssertionError("Graph missing targets 'y'")

    print("Data processing verified.")


def test_model_forward_pass():
    """
    Verifies that the model can perform a forward pass on a batch of data.
    """
    print("\n--- Testing Model Forward Pass ---")
    # Get dataloaders (force processing by setting load_cached_data=False for this specific test,
    # though in the main run we will use cache)
    # We use the demo metadata paths set in Config
    train_loader, _, _ = get_dataloaders(load_cached_data=False, batch_size=2)

    # Get one batch
    batch = next(iter(train_loader))
    batch = batch.to(Config.device)

    # Instantiate model
    model = IS_RA_CGN(Config).to(Config.device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(batch)

    print(f"Batch size: {batch.num_graphs}")
    print(f"Output shape: {output.shape}")

    if output.shape != (batch.num_graphs, 2):
        raise AssertionError(
            f"Expected output shape ({batch.num_graphs}, 2), got {output.shape}"
        )

    print("Model forward pass verified.")


def run_full_demo():
    """
    Runs the full training and submission generation pipeline using the demo configuration.
    """
    print("\n--- Running Full Training Demo ---")

    # Clear any existing cache in the demo directory to ensure fresh processing
    if os.path.exists(Config.cache_dir):
        shutil.rmtree(Config.cache_dir)
    os.makedirs(Config.cache_dir, exist_ok=True)

    # Run training
    # This will:
    # 1. Process data from demo metadata and save to cache
    # 2. Train for Config.num_epochs (2)
    # 3. Save best model
    # 4. Generate submission for the demo test set
    run_training()

    # Verify submission file
    submission_path = os.path.join(Config.submission_dir, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # We expect 10 rows because our demo test metadata has 10 rows
    if len(sub_df) != 10:
        raise AssertionError(f"Expected 10 rows in submission, found {len(sub_df)}")

    print("Full demo pipeline completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Component Unit Tests
        test_target_scaler()
        test_data_processing()
        test_model_forward_pass()

        # 3. Integration Test
        run_full_demo()

        print("\nAll tests passed!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise e
