import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from functools import partial

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable tqdm globally to comply with "no progress bars" requirement
from tqdm import tqdm


def noop_tqdm(*args, **kwargs):
    if args:
        return args[0]
    return kwargs.get("iterable", [])


sys.modules["tqdm"].tqdm = noop_tqdm

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Import Config and modify it for a fast demo run
from library.config import Config

# Monkey-patch Config for speed and debugging
print("Configuring environment for demo run...")
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 molecules
Config.MAX_EPOCHS = 2  # Run only 2 epochs
Config.HIDDEN_DIM = 32  # Small hidden dimension
Config.NUM_LAYERS = 2  # Fewer GNN layers
Config.NUM_HEADS = 2
Config.NUM_RBF = 16
Config.NUM_ANGLE_RBF = 8
Config.BATCH_SIZE = 16
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
Config.WORKING_DIR = "./working/demo_run"
Config.PROCESSED_CACHE_DIR = os.path.join(Config.WORKING_DIR, "processed")
Config.STATS_PATH = os.path.join(Config.WORKING_DIR, "target_stats.npy")
Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
os.makedirs(Config.PROCESSED_CACHE_DIR, exist_ok=True)

# Set seeds
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# Import remaining library modules
from library.data import get_datasets
from library.model import DualGraphNetwork
from library.engine import Trainer
from library.utils import GroupScaler, mean_log_mae
from torch_geometric.loader import DataLoader


def run_demo():
    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n--- Step 1: Data Loading ---")
    # Initialize datasets (this triggers processing if not cached)
    # We force reprocessing to ensure our DEBUG settings take effect
    if os.path.exists(Config.PROCESSED_CACHE_DIR):
        import shutil

        shutil.rmtree(Config.PROCESSED_CACHE_DIR)
    os.makedirs(Config.PROCESSED_CACHE_DIR, exist_ok=True)

    train_ds, val_ds, test_ds = get_datasets(load_cached_data=False)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    # Verify dataset is not empty
    assert len(train_ds) > 0, "Train dataset is empty!"
    assert len(val_ds) > 0, "Val dataset is empty!"

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Inspect a single batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys}")

    # Assertions for batch structure
    assert hasattr(batch, "x"), "Batch missing node features 'x'"
    assert hasattr(batch, "edge_index"), "Batch missing 'edge_index'"
    assert hasattr(batch, "coupling_value"), "Batch missing targets 'coupling_value'"
    assert (
        batch.x.shape[1] == len(batch.x.shape) - 1
    ), "Node features should be 1D indices or 2D [N, F]"

    # ==========================================
    # 3. Model Initialization & Verification
    # ==========================================
    print("\n--- Step 2: Model Initialization ---")
    device = torch.device(Config.DEVICE)
    model = DualGraphNetwork().to(device)

    # Move batch to device for forward pass check
    batch = batch.to(device)

    # Forward pass
    with torch.no_grad():
        pred_c, pred_s, pred_m = model(batch)

    print("Forward pass successful.")
    print(f"Coupling Prediction Shape: {pred_c.shape}")
    print(f"Shielding Prediction Shape: {pred_s.shape}")
    print(f"Charge Prediction Shape: {pred_m.shape}")

    # Assertions for output shapes
    assert (
        pred_c.shape == batch.coupling_value.shape
    ), "Coupling prediction shape mismatch"
    assert pred_m.shape == (batch.num_atoms, 1), "Charge prediction shape mismatch"
    # Shielding is 9 components per atom
    assert pred_s.shape == (batch.num_atoms, 9), "Shielding prediction shape mismatch"

    # ==========================================
    # 4. Utility Logic Verification
    # ==========================================
    print("\n--- Step 3: Utility Verification ---")

    # Test GroupScaler
    scaler = GroupScaler()
    # Create dummy data: Type 'A' has mean 10, Type 'B' has mean 20
    dummy_vals = np.array([10.0, 10.0, 20.0, 20.0])
    dummy_types = np.array(["A", "A", "B", "B"])
    df_dummy = pd.DataFrame({"val": dummy_vals, "type": dummy_types})

    scaler.fit(df_dummy, "val", "type")
    assert scaler.means["A"] == 10.0
    assert scaler.means["B"] == 20.0

    # Transform (should become 0s if std is handled, std of constant is usually nan or 0)
    # The scaler implementation handles std=0 by setting it to 1.0
    transformed = scaler.transform(dummy_vals, dummy_types)
    expected = np.array([0.0, 0.0, 0.0, 0.0])
    assert np.allclose(
        transformed, expected
    ), f"Scaler transform failed. Got {transformed}"

    # Test Metric (Log MAE)
    # Perfect prediction -> log(1e-9) approx -20.7
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    types = np.array(["1JHC", "1JHC"])
    score = mean_log_mae(y_true, y_pred, types)
    print(f"Perfect score (LogMAE): {score:.4f}")
    assert score < -10.0, "Metric calculation for perfect prediction is incorrect"

    # ==========================================
    # 5. Training Loop (Trainer)
    # ==========================================
    print("\n--- Step 4: Training Loop ---")
    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run training
    # Since we set MAX_EPOCHS=2 and DEBUG=True, this should be very fast
    trainer.run()

    # Verify model file creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Best model file was not saved!"
    print("Training completed and model saved.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n--- Step 5: Inference & Submission ---")
    trainer.predict_and_submit()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {list(df_sub.columns)}")

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        "scalar_coupling_constant" in df_sub.columns
    ), "Submission missing target column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if IDs match test set (in debug mode, test set is small)
    # We need to load test metadata to compare
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)
    # In debug mode, the dataset filters molecules, so we only check if submission IDs
    # are a subset of valid IDs or match the filtered test set count.
    # Since Trainer iterates the DataLoader which respects the filtered dataset,
    # the submission length should match the test_ds length * couplings per molecule.

    # Count total couplings in test_ds
    total_test_couplings = 0
    for data in test_loader:
        total_test_couplings += data.coupling_id.numel()

    assert (
        len(df_sub) == total_test_couplings
    ), f"Submission row count ({len(df_sub)}) does not match test loader count ({total_test_couplings})"

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demo()
