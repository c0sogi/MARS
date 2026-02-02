import os
import sys
import torch
import numpy as np
import shutil
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_factory import DataFactory
from library.loader import FlattenedGraphDataset, GraphCollator
from library.model import DirectionalMPNN
from library.engine import Trainer


# ==========================================
# 1. Configuration & Setup
# ==========================================
def setup_demo_config():
    """
    Overrides default configuration for a fast demonstration run.
    """
    print("[Demo] Configuring environment...")

    # Enable Debug Mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Process only 100 molecules

    # Training Hyperparameters for Speed
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect output paths to a demo directory
    Config.WORKING_DIR = "./working/demo_run"

    # IMPORTANT: Since PROCESSED_DATA_DIR is defined at class level based on the original WORKING_DIR,
    # we must explicitly update it after changing WORKING_DIR.
    Config.PROCESSED_DATA_DIR = os.path.join(Config.WORKING_DIR, "processed")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(42)
    print(f"[Demo] Working Directory: {Config.WORKING_DIR}")


# ==========================================
# 2. Data Processing Demonstration
# ==========================================
def demo_data_processing():
    print("\n[Demo] --- Step 1: Data Processing (DataFactory) ---")

    factory = DataFactory()

    # Process training data from scratch (load_cached_data=False)
    # This reads metadata, structures, and aux files, then creates numpy arrays
    print("[Demo] Processing training split...")
    data = factory.process_dataset(split="train", load_cached_data=False)

    # Verify Data Integrity
    expected_keys = [
        "node_x",
        "node_pos",
        "edge_index",
        "edge_attr",
        "triplet_index",
        "triplet_attr",
        "coupling_value",
        "coupling_type",
        "coupling_edge_index",
    ]

    for key in expected_keys:
        if key not in data:
            raise AssertionError(f"Missing key in processed data: {key}")

    # Check shapes
    num_nodes = len(data["node_x"])
    num_edges = data["edge_index"].shape[1]

    print(f"[Demo] Processed {num_nodes} nodes and {num_edges} edges.")

    if len(data["node_pos"]) != num_nodes:
        raise AssertionError("Mismatch between node_x and node_pos lengths.")

    # Check if stats file was created
    stats_path = os.path.join(Config.PROCESSED_DATA_DIR, "stats.npy")
    if not os.path.exists(stats_path):
        raise AssertionError("Statistics file was not created by DataFactory.")

    print("[Demo] Data processing successful. Stats saved.")
    return data


# ==========================================
# 3. Data Loading Demonstration
# ==========================================
def demo_data_loading():
    print("\n[Demo] --- Step 2: Data Loading (Dataset & Collator) ---")

    # Initialize Dataset
    # This wraps the processed numpy arrays and handles slicing per molecule
    dataset = FlattenedGraphDataset(split="train", load_cached_data=True)

    print(f"[Demo] Dataset contains {len(dataset)} molecules.")
    if len(dataset) == 0:
        raise AssertionError("Dataset is empty.")

    # Test getting a single item
    sample = dataset[0]
    print(f"[Demo] Sample 0 keys: {list(sample.keys())}")

    if "node_x" not in sample or "edge_index" not in sample:
        raise AssertionError("Sample missing required graph keys.")

    # Test Batching with Collator
    collator = GraphCollator()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))

    print(f"[Demo] Batch keys: {list(batch.keys())}")

    # Verify Batch Shapes
    # node_batch should map nodes to batch index (0 to 3)
    if batch["node_batch"].max() >= 4:
        raise AssertionError("Batch indices exceed batch size.")

    if batch["edge_index"].shape[0] != 2:
        raise AssertionError("edge_index should have shape (2, M).")

    print("[Demo] Data loading and batching successful.")
    return batch


# ==========================================
# 4. Model Demonstration
# ==========================================
def demo_model_inference(batch):
    print("\n[Demo] --- Step 3: Model Architecture (DirectionalMPNN) ---")

    device = torch.device(Config.DEVICE)
    model = DirectionalMPNN().to(device)

    # Move batch to device
    batch_gpu = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch_gpu[k] = v.to(device)

    print("[Demo] Running forward pass...")
    with torch.no_grad():
        preds = model(batch_gpu)

    # Verify Outputs
    print(f"[Demo] Prediction keys: {list(preds.keys())}")

    if "coupling" not in preds:
        raise AssertionError("Model failed to output 'coupling' predictions.")

    # Check output shape matches target shape
    target_shape = batch_gpu["coupling_value"].shape
    pred_shape = preds["coupling"].view(-1).shape

    print(f"[Demo] Target shape: {target_shape}, Pred shape: {pred_shape}")

    if pred_shape != target_shape:
        # Note: Model output is (N, 1), target is (N,). View fix is applied in loss usually.
        # Here we just check the number of elements matches.
        if preds["coupling"].numel() != batch_gpu["coupling_value"].numel():
            raise AssertionError("Prediction size mismatch.")

    print("[Demo] Model forward pass successful.")


# ==========================================
# 5. Training Loop Demonstration
# ==========================================
def demo_training():
    print("\n[Demo] --- Step 4: Training Loop (Trainer) ---")

    # Initialize Trainer
    # This handles Model, Optimizer, Scheduler, and Data Loaders internally
    trainer = Trainer(load_cached_data=True)

    # Run Training
    print("[Demo] Starting fit() for 2 epochs...")
    trainer.fit()

    # Verify Model Checkpoint
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise AssertionError("Model checkpoint was not saved after training.")

    print(f"[Demo] Training complete. Best model saved to {Config.MODEL_SAVE_PATH}")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Process Data
        # We need to process 'val' as well for the Trainer to work,
        # as Trainer initializes both train and val loaders.
        print("\n[Demo] Pre-processing data for Train and Val splits...")
        factory = DataFactory()
        factory.process_dataset(split="train", load_cached_data=False)
        factory.process_dataset(split="val", load_cached_data=False)

        # 3. Demonstrate individual components
        # Verify DataFactory output structure
        data_dict = factory.process_dataset(split="train", load_cached_data=True)

        # Verify Loader and Batching
        batch = demo_data_loading()

        # Verify Model
        demo_model_inference(batch)

        # 4. Run Integration Test (Trainer)
        demo_training()

        print("\n[Demo] All demonstrations completed successfully.")

    except Exception as e:
        print(f"\n[Demo] FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
