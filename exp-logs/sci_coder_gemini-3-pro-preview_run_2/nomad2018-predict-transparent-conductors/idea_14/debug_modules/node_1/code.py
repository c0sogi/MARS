import os
import sys
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed, StandardScaler, rmsle
from library.data import get_loaders, get_pbc_graph
from library.model import CGCNN_IB
from library.train import train_one_epoch, validate, generate_submission


def create_demo_metadata(n_samples=20):
    """
    Creates smaller metadata files in the working directory to speed up the demo.
    """
    print(f"Creating demo metadata with {n_samples} samples per split...")

    # Define demo paths
    demo_meta_dir = os.path.join(Config.WORKING_DIR, "demo_metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Read original metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    # Sample subset
    train_subset = train_df.head(n_samples)
    val_subset = val_df.head(n_samples)
    test_subset = test_df.head(n_samples)

    # Save subsets
    train_path = os.path.join(demo_meta_dir, "train_metadata.csv")
    val_path = os.path.join(demo_meta_dir, "val_metadata.csv")
    test_path = os.path.join(demo_meta_dir, "test_metadata.csv")

    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def configure_demo_settings(train_path, val_path, test_path):
    """
    Overrides Config attributes to use demo paths and reduce runtime.
    """
    print("Configuring demo settings...")

    # Override paths
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    # Use a separate cache directory for demo
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.TRAIN_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    Config.VAL_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    Config.TEST_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "test_graphs.npz")
    Config.TARGET_SCALER_PATH = os.path.join(Config.CACHE_DIR, "target_scaler.npz")

    # Override checkpoints and submission
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "demo_checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "demo_submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Re-run setup to create new directories
    Config.setup()


def test_data_loading():
    print("\n--- Testing Data Loading ---")
    # Load data (force processing from scratch by setting load_cached_data=False)
    train_loader, val_loader, test_loader, scaler = get_loaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify a batch
    batch = next(iter(train_loader))
    print(f"Batch structure: {batch}")

    assert batch.x.dim() == 1, "Node features should be 1D (atomic numbers)"
    assert batch.edge_index.shape[0] == 2, "Edge index should have shape (2, E)"
    assert batch.edge_attr.shape[1] == Config.NUM_RBF_BINS, "Edge attr dim mismatch"
    assert batch.y.shape[1] == 2, "Target shape mismatch (should be 2 columns)"

    # Check scaler
    assert scaler is not None, "Scaler should be initialized"
    assert scaler.mean is not None, "Scaler mean should be computed"

    return train_loader, val_loader, test_loader, scaler


def test_model_architecture(device):
    print("\n--- Testing Model Architecture ---")
    model = CGCNN_IB(config=Config).to(device)
    print(model)

    # Create dummy data
    # 5 nodes, 10 edges
    dummy_x = torch.randint(0, 100, (5,)).to(device)
    dummy_edge_index = torch.randint(0, 5, (2, 10)).to(device)
    dummy_edge_attr = torch.randn(10, Config.NUM_RBF_BINS).to(device)
    dummy_batch = torch.zeros(5, dtype=torch.long).to(device)

    from torch_geometric.data import Data, Batch

    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr)
    dummy_batch_obj = Batch.from_data_list([dummy_data])

    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch_obj)

    print(f"Model output shape: {out.shape}")
    assert out.shape == (1, 2), f"Expected output shape (1, 2), got {out.shape}"

    return model


def test_training_loop(model, train_loader, val_loader, device, scaler):
    print("\n--- Testing Training Loop ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # Train for a few epochs
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device, scaler)
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Val RMSLE={val_score:.4f}")

        assert not np.isnan(loss), "Training loss is NaN"
        assert not np.isnan(val_score), "Validation score is NaN"

    # Save checkpoint
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Checkpoint not saved"
    print("Training loop completed and model saved.")


def test_inference(model, test_loader, device, scaler):
    print("\n--- Testing Inference and Submission ---")

    # Load model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Generate submission
    generate_submission(model, test_loader, device, scaler, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify submission format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df.shape}")
    print(f"Submission columns: {df.columns.tolist()}")

    assert len(df) == 20, f"Expected 20 rows (from demo subset), got {len(df)}"
    assert "id" in df.columns
    assert "formation_energy_ev_natom" in df.columns
    assert "bandgap_energy_ev" in df.columns

    # Check values are reasonable (not NaN or Inf)
    assert not df.isnull().values.any(), "Submission contains NaNs"

    print("Inference test passed.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Create subset metadata for speed
    train_path, val_path, test_path = create_demo_metadata(n_samples=20)
    configure_demo_settings(train_path, val_path, test_path)

    # 3. Test Data Loading
    train_loader, val_loader, test_loader, scaler = test_data_loading()

    # 4. Test Model
    model = test_model_architecture(device)

    # 5. Test Training
    test_training_loop(model, train_loader, val_loader, device, scaler)

    # 6. Test Inference
    test_inference(model, test_loader, device, scaler)

    print("\nAll tests passed successfully!")
