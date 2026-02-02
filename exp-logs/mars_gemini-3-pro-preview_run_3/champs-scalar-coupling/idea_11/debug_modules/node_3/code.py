import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import logging

# Ensure warnings are suppressed for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import setup_logger
from library.data_prep import DataProcessor
from library.dataset import MolecularGraphDataset
from library.model import ScalarCouplingModel
from library.runner import Trainer


def create_truncated_dataset(source_dir, dest_dir, n_train=50, n_val=20, n_test=20):
    """
    Creates a tiny subset of the dataset for demonstration purposes.
    """
    print(f"Creating truncated dataset in {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)

    # 1. Load Metadata and Sample Molecules
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    # Get unique molecules
    train_mols = train_meta["molecule_name"].unique()[:n_train]
    val_mols = val_meta["molecule_name"].unique()[:n_val]
    test_mols = test_meta["molecule_name"].unique()[:n_test]

    selected_mols = np.concatenate([train_mols, val_mols, test_mols])

    # Filter Metadata
    train_sub = train_meta[train_meta["molecule_name"].isin(train_mols)].copy()
    val_sub = val_meta[val_meta["molecule_name"].isin(val_mols)].copy()
    test_sub = test_meta[test_meta["molecule_name"].isin(test_mols)].copy()

    # Save Metadata
    train_sub.to_csv(os.path.join(dest_dir, "train_metadata.csv"), index=False)
    val_sub.to_csv(os.path.join(dest_dir, "val_metadata.csv"), index=False)
    test_sub.to_csv(os.path.join(dest_dir, "test_metadata.csv"), index=False)

    # 2. Filter Structures
    print("  Filtering structures...")
    df_struct = pd.read_csv(Config.STRUCTURES_CSV)
    df_struct_sub = df_struct[df_struct["molecule_name"].isin(selected_mols)].copy()
    df_struct_sub.to_csv(os.path.join(dest_dir, "structures.csv"), index=False)

    # 3. Filter Aux Data
    print("  Filtering auxiliary files...")
    # Charges
    df_charge = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)
    df_charge_sub = df_charge[df_charge["molecule_name"].isin(selected_mols)].copy()
    df_charge_sub.to_csv(os.path.join(dest_dir, "mulliken_charges.csv"), index=False)

    # Shielding
    df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
    df_shield_sub = df_shield[df_shield["molecule_name"].isin(selected_mols)].copy()
    df_shield_sub.to_csv(
        os.path.join(dest_dir, "magnetic_shielding_tensors.csv"), index=False
    )

    # Potential Energy (optional, not strictly used by DataProcessor but good to have)
    df_pot = pd.read_csv(Config.POTENTIAL_ENERGY_CSV)
    df_pot_sub = df_pot[df_pot["molecule_name"].isin(selected_mols)].copy()
    df_pot_sub.to_csv(os.path.join(dest_dir, "potential_energy.csv"), index=False)

    # Dipole Moments (optional)
    df_dip = pd.read_csv(Config.DIPOLE_MOMENTS_CSV)
    df_dip_sub = df_dip[df_dip["molecule_name"].isin(selected_mols)].copy()
    df_dip_sub.to_csv(os.path.join(dest_dir, "dipole_moments.csv"), index=False)

    return train_sub, val_sub, test_sub


def patch_config(temp_dir):
    """
    Modifies the Config class to point to the temporary truncated dataset
    and adjusts hyperparameters for speed.
    """
    print("Patching Configuration...")

    # Paths
    Config.WORKING_DIR = os.path.join(temp_dir, "working")
    Config.PROCESSED_DATA_DIR = os.path.join(Config.WORKING_DIR, "processed_soa")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Input Files
    Config.STRUCTURES_CSV = os.path.join(temp_dir, "structures.csv")
    Config.MULLIKEN_CHARGES_CSV = os.path.join(temp_dir, "mulliken_charges.csv")
    Config.MAGNETIC_SHIELDING_CSV = os.path.join(
        temp_dir, "magnetic_shielding_tensors.csv"
    )
    Config.POTENTIAL_ENERGY_CSV = os.path.join(temp_dir, "potential_energy.csv")
    Config.DIPOLE_MOMENTS_CSV = os.path.join(temp_dir, "dipole_moments.csv")

    # Metadata Files
    Config.TRAIN_META_PATH = os.path.join(temp_dir, "train_metadata.csv")
    Config.VAL_META_PATH = os.path.join(temp_dir, "val_metadata.csv")
    Config.TEST_META_PATH = os.path.join(temp_dir, "test_metadata.csv")

    # Cache Paths (Update based on new PROCESSED_DATA_DIR)
    Config.CACHE_NODES_PATH = os.path.join(Config.PROCESSED_DATA_DIR, "nodes.npy")
    Config.CACHE_COORDS_PATH = os.path.join(Config.PROCESSED_DATA_DIR, "coords.npy")
    Config.CACHE_EDGES_PATH = os.path.join(
        Config.PROCESSED_DATA_DIR, "edge_indices.npy"
    )
    Config.CACHE_EDGE_ATTRS_PATH = os.path.join(
        Config.PROCESSED_DATA_DIR, "edge_attrs.npy"
    )
    Config.CACHE_TRIPLETS_PATH = os.path.join(Config.PROCESSED_DATA_DIR, "triplets.npy")
    Config.CACHE_MOL_INDICES_PATH = os.path.join(
        Config.PROCESSED_DATA_DIR, "mol_indices.npy"
    )
    Config.CACHE_TRAIN_TARGETS_PATH = os.path.join(
        Config.PROCESSED_DATA_DIR, "train_targets.npy"
    )
    Config.CACHE_VAL_TARGETS_PATH = os.path.join(
        Config.PROCESSED_DATA_DIR, "val_targets.npy"
    )
    Config.STATS_PATH = os.path.join(Config.PROCESSED_DATA_DIR, "stats.npy")

    # Model/Output Paths
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for Demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 1000  # Large enough to cover our truncated set
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2
    Config.HIDDEN_DIM = 64  # Reduce model size for speed
    Config.NUM_LAYERS = 2


def run_demo():
    # Setup directories
    base_temp_dir = "./working/demo_execution"
    if os.path.exists(base_temp_dir):
        shutil.rmtree(base_temp_dir)
    os.makedirs(base_temp_dir, exist_ok=True)

    # 1. Create Truncated Data
    create_truncated_dataset(
        source_dir="./input", dest_dir=base_temp_dir, n_train=50, n_val=20, n_test=20
    )

    # 2. Patch Config
    patch_config(base_temp_dir)

    # Set Seed
    Config.set_seed()

    # 3. Data Processing
    print("\n--- Running Data Processing ---")
    processor = DataProcessor()
    # Force processing by ensuring cache doesn't exist (handled by new dir)
    data_dict = processor.process_all(load_cached_data=True)

    # Verification
    assert data_dict is not None
    assert "nodes" in data_dict
    assert "coords" in data_dict
    print("Data processing completed successfully.")

    # 4. Dataset Loading & Verification
    print("\n--- Verifying Dataset ---")
    train_ds = MolecularGraphDataset(split="train")
    val_ds = MolecularGraphDataset(split="val")

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")

    assert len(train_ds) > 0
    assert len(val_ds) > 0

    # Check a sample
    sample = train_ds[0]
    print(f"Sample Graph: {sample}")
    print(f"  Nodes: {sample.x.shape}")
    print(f"  Edges: {sample.edge_index.shape}")
    print(f"  Triplets: {sample.triplets.shape}")
    print(f"  Couplings: {sample.y.shape}")

    # Assertions
    assert sample.x.dim() == 1
    assert sample.edge_index.shape[0] == 2
    assert sample.triplets.shape[0] == 2
    assert sample.y.dim() == 1
    assert sample.aux_shielding.shape[1] == 9

    # 5. Model Verification
    print("\n--- Verifying Model ---")
    model = ScalarCouplingModel().to(Config.DEVICE)

    # Create a batch
    from torch_geometric.loader import DataLoader

    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    batch = next(iter(loader)).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        pred_c, pred_s, pred_q = model(batch)

    print(f"Predictions shape: {pred_c.shape}")
    print(f"Targets shape: {batch.y.shape}")

    assert pred_c.shape == batch.y.shape
    assert pred_s.shape == batch.aux_shielding.shape
    assert pred_q.shape == batch.aux_charge.shape

    print("Model forward pass successful.")

    # 6. Training Loop
    print("\n--- Starting Training Demo ---")
    trainer = Trainer()
    trainer.fit()

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training completed and model saved.")

    # 7. Prediction/Inference
    print("\n--- Running Inference ---")
    trainer.predict()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    assert "id" in df_sub.columns
    assert "scalar_coupling_constant" in df_sub.columns
    assert len(df_sub) > 0

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
