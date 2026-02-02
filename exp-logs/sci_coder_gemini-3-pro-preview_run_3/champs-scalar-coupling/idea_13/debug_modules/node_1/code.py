import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings
import time

# Import provided library modules
from library.config import Config
from library.data_processor import DataProcessor
from library.dataset import SoADataset, SoACollator
from library.model import DMPNN
from library.losses import LossComputer
from library.trainer import Trainer
from library.inference import Predictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Overrides the default Config to run a fast, lightweight demo.
    """
    print(">>> Setting up Demo Configuration...")

    # Enable Debug Mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Only process 100 molecules

    # Training settings for speed
    Config.MAX_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Redirect output to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.PROCESSED_DIR = os.path.join(Config.WORKING_DIR, "processed")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.STATS_FILE = os.path.join(Config.PROCESSED_DIR, "stats.npy")

    # Manually create these directories since Config creates them at import time
    # and we just changed the paths
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    Config.set_seed(42)
    print("    Configuration updated for demo run.")


def demo_data_processing():
    """
    Demonstrates the ETL pipeline using DataProcessor.
    """
    print("\n>>> 1. Running Data Processor...")
    processor = DataProcessor()

    # Run the pipeline (force reload to ignore any existing cache)
    processor.run(load_cached_data=False)

    # Verification
    expected_files = [
        "train_pos.npy",
        "train_edge_index.npy",
        "train_triplet_index.npy",
        "train_coupling_value.npy",
        "stats.npy",
        "completed.flag",
    ]

    print("    Verifying output files...")
    for fname in expected_files:
        fpath = os.path.join(Config.PROCESSED_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected processed file missing: {fpath}")

    # Check stats file content
    stats = np.load(Config.STATS_FILE, allow_pickle=True).item()
    assert isinstance(stats, dict), "Stats file should contain a dictionary."
    print("    Data Processing successful. Files created.")


def demo_dataset_and_collator():
    """
    Demonstrates loading the SoADataset and batching with SoACollator.
    """
    print("\n>>> 2. Testing Dataset and Collator...")

    # Load dataset
    dataset = SoADataset(split="train", load_cached_data=True)
    print(f"    Loaded Train Dataset. Molecules: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check DataProcessor logic.")

    # Create Collator
    collator = SoACollator(dataset)

    # Simulate a DataLoader batch
    batch_mol_ids = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collator(batch_mol_ids)

    # Verification of Batch Shapes
    print("    Verifying batch shapes...")

    # Nodes
    num_nodes = batch["pos"].shape[0]
    assert batch["pos"].shape == (num_nodes, 3)
    assert batch["node_type"].shape == (num_nodes,)
    assert batch["batch"].shape == (num_nodes,)

    # Edges
    num_edges = batch["edge_index"].shape[1]
    assert batch["edge_index"].shape == (2, num_edges)
    assert batch["edge_vec"].shape == (num_edges, 3)

    # Couplings (Targets)
    num_couplings = batch["coupling_value"].shape[0]
    assert batch["coupling_atom_index"].shape == (2, num_couplings)
    assert batch["coupling_value"].shape == (num_couplings,)

    # Check connectivity validity
    if num_edges > 0:
        assert batch["edge_index"].max() < num_nodes, "Edge index out of bounds."
    if num_couplings > 0:
        assert (
            batch["coupling_atom_index"].max() < num_nodes
        ), "Coupling index out of bounds."

    print("    Batch structure verified.")
    return batch


def demo_model_forward(batch):
    """
    Demonstrates the DMPNN model initialization and forward pass.
    """
    print("\n>>> 3. Testing Model Forward Pass...")

    device = torch.device("cpu")  # Use CPU for simple demo check

    model = DMPNN(
        hidden_dim=32, num_layers=2, num_rbf=16, num_angle_rbf=8  # Small dim for speed
    ).to(device)

    # Move batch to device
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.to(device)
        else:
            batch_dev[k] = v

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(batch_dev)

    # Verification
    print("    Verifying prediction shapes...")
    num_couplings = batch["coupling_value"].shape[0]
    num_nodes = batch["pos"].shape[0]

    assert "coupling" in preds
    assert preds["coupling"].shape == (num_couplings,)

    assert "charge" in preds
    assert preds["charge"].shape == (num_nodes,)

    assert "shielding" in preds
    assert preds["shielding"].shape == (num_nodes, 9)

    print("    Model forward pass successful.")
    return model, preds, batch_dev


def demo_loss_computation(preds, batch):
    """
    Demonstrates LossComputer logic.
    """
    print("\n>>> 4. Testing Loss Computation...")

    loss_computer = LossComputer()

    # Compute Loss
    total_loss, components = loss_computer(preds, batch)

    # Compute Metric
    metric = loss_computer.compute_metric(preds, batch)

    # Verification
    print(f"    Total Loss: {total_loss.item():.4f}")
    print(f"    Metric (LogMAE): {metric:.4f}")

    assert total_loss.item() > 0, "Loss should be positive."
    assert "loss_coupling" in components
    assert isinstance(metric, float)

    print("    Loss computation verified.")


def demo_full_training():
    """
    Demonstrates the Trainer class running a full (short) cycle.
    """
    print("\n>>> 5. Running Trainer (1 Epoch)...")

    trainer = Trainer()

    # Run training
    # Config is set to 1 epoch, debug mode
    trainer.train()

    # Verify checkpoint creation
    # Note: Trainer constructs path using CHECKPOINT_DIR
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Trainer failed to save best_model.pth")

    print("    Training cycle complete. Checkpoint saved.")


def demo_inference():
    """
    Demonstrates the Predictor class generating a submission.
    """
    print("\n>>> 6. Running Inference...")

    predictor = Predictor()

    # Run prediction
    predictor.predict(batch_size=4, num_workers=0)

    # Verify submission file
    sub_path = Config.SUBMISSION_PATH
    if not os.path.exists(sub_path):
        raise FileNotFoundError("Predictor failed to create submission.csv")

    df = pd.read_csv(sub_path)
    print(f"    Submission generated. Rows: {len(df)}")

    assert "id" in df.columns
    assert "scalar_coupling_constant" in df.columns
    assert not df.isnull().values.any(), "Submission contains NaNs."

    print("    Inference successful.")


if __name__ == "__main__":
    t_start = time.time()

    # 1. Setup
    setup_demo_environment()

    # 2. Data Processing
    demo_data_processing()

    # 3. Dataset & Collator
    batch = demo_dataset_and_collator()

    # 4. Model
    model, preds, batch_dev = demo_model_forward(batch)

    # 5. Loss
    demo_loss_computation(preds, batch_dev)

    # 6. Training Loop
    demo_full_training()

    # 7. Inference
    demo_inference()

    print(f"\n>>> Demo Completed Successfully in {time.time() - t_start:.2f} seconds.")
