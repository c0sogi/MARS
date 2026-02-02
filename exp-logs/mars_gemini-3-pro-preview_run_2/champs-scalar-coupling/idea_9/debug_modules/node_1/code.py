import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from torch_geometric.loader import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_target_stats, denormalize_predictions
from library.features import RadialBasisFunctions, SphericalBasisFunctions
from library.dataset import get_molecular_data
from library.model import HGANet
from library.engine import train_one_epoch, evaluate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo...")

    # Override Config for a fast, lightweight run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Only process 50 molecules
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_TRAIN_PATH = os.path.join(Config.WORKING_DIR, "cached_train_demo.npz")
    Config.CACHE_VAL_PATH = os.path.join(Config.WORKING_DIR, "cached_val_demo.npz")

    # Reduce Model Complexity for speed
    Config.HIDDEN_DIM = 32
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.NUM_RBF = 10
    Config.NUM_SBF = 3

    # Training settings
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Verify Feature Engineering Components
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Feature Engineering (RBF/SBF)...")

    # Test RBF
    rbf = RadialBasisFunctions(cutoff=5.0, num_rbf=Config.NUM_RBF)
    dummy_dist = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rbf_out = rbf(dummy_dist)

    assert rbf_out.shape == (
        4,
        Config.NUM_RBF,
    ), f"RBF output shape mismatch. Expected (4, {Config.NUM_RBF}), got {rbf_out.shape}"
    print("    RadialBasisFunctions: OK")

    # Test SBF
    sbf = SphericalBasisFunctions(
        cutoff=5.0, num_rbf=Config.NUM_RBF, num_sbf=Config.NUM_SBF
    )
    dummy_angle = torch.tensor([0.5, 1.0, 1.5, 2.0])  # Radians
    sbf_out = sbf(dummy_dist, dummy_angle)

    expected_sbf_dim = Config.NUM_RBF * Config.NUM_SBF
    assert sbf_out.shape == (
        4,
        expected_sbf_dim,
    ), f"SBF output shape mismatch. Expected (4, {expected_sbf_dim}), got {sbf_out.shape}"
    print("    SphericalBasisFunctions: OK")

    # --------------------------------------------------------------------------
    # 3. Data Loading & Processing
    # --------------------------------------------------------------------------
    print("\n[3] Loading and Processing Data (Debug Subset)...")

    # Force reload from scratch to demonstrate processing logic
    if os.path.exists(Config.CACHE_TRAIN_PATH):
        os.remove(Config.CACHE_TRAIN_PATH)

    train_dataset = get_molecular_data(split="train", load_cached_data=False)

    assert len(train_dataset) > 0, "Dataset is empty!"
    assert len(train_dataset) <= Config.DEBUG_SUBSET_SIZE, "Dataset subsetting failed!"

    # Inspect one sample
    sample = train_dataset[0]
    print(f"    Loaded {len(train_dataset)} molecules.")
    print(f"    Sample Molecule: {sample.molecule_name}")
    print(f"    Num Atoms: {sample.x.size(0)}")
    print(f"    Num Edges: {sample.edge_index.size(1)}")
    print(f"    Num Targets: {sample.target_val.size(0)}")

    # Verify Data attributes
    required_keys = [
        "x",
        "edge_index",
        "edge_attr",
        "triplet_index",
        "triplet_attr",
        "target_val",
        "target_type",
    ]
    for key in required_keys:
        assert hasattr(sample, key), f"Missing key in Data object: {key}"

    print("    Data Structure: OK")

    # --------------------------------------------------------------------------
    # 4. Target Statistics
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Target Statistics...")

    # We need to load the metadata df to compute stats (or load cached)
    # Since we ran get_molecular_data, stats should be cached in working dir
    # Note: get_molecular_data uses the full metadata file unless we manually filter it,
    # but in debug mode inside get_molecular_data, it filters the DF.
    # Let's explicitly compute stats on the debug subset for consistency in this demo.

    df_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    molecules = df_meta["molecule_name"].unique()[: Config.DEBUG_SUBSET_SIZE]
    df_subset = df_meta[df_meta["molecule_name"].isin(molecules)]

    stats = get_target_stats(df_subset, load_cached_data=False)

    assert len(stats) == len(Config.COUPLING_TYPES), "Stats dictionary missing types."
    print(f"    Stats computed for {len(stats)} coupling types.")
    print(f"    Type 0 (1JHC) Mean: {stats[0][0]:.4f}, Std: {stats[0][1]:.4f}")

    # --------------------------------------------------------------------------
    # 5. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[5] Initializing HGANet Model...")

    model = HGANet().to(Config.DEVICE)

    # Basic parameter check
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Model instantiated with {num_params:,} parameters.")

    # Forward pass check with a single batch
    loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch = next(iter(loader)).to(Config.DEVICE)

    with torch.no_grad():
        out = model(batch)

    assert (
        out.shape == batch.target_val.shape
    ), f"Model output shape {out.shape} does not match target shape {batch.target_val.shape}"
    print("    Forward pass: OK")

    # --------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    avg_loss = train_one_epoch(model, loader, optimizer, criterion, Config.DEVICE)

    assert not np.isnan(avg_loss), "Training loss is NaN!"
    assert avg_loss > 0, "Training loss should be positive."
    print(f"    Epoch 1 Loss: {avg_loss:.6f}")

    # --------------------------------------------------------------------------
    # 7. Evaluation Demonstration
    # --------------------------------------------------------------------------
    print("\n[7] Running Evaluation...")

    # Evaluate on the same training set just for demonstration
    metric = evaluate(model, loader, Config.DEVICE, stats)

    # Metric is Log MAE, can be negative
    print(f"    Evaluation Metric (LMAE): {metric:.6f}")
    assert isinstance(metric, float), "Metric should be a float."

    # --------------------------------------------------------------------------
    # 8. Inference & Denormalization
    # --------------------------------------------------------------------------
    print("\n[8] Inference and Denormalization...")

    model.eval()
    with torch.no_grad():
        # Predict on one batch
        preds_norm = model(batch).cpu().numpy()
        types = batch.target_type.cpu().numpy()

        # Denormalize
        preds_real = denormalize_predictions(preds_norm, types, stats)

        # Check logic
        # Pick first item
        t_idx = types[0]
        m, s = stats[t_idx]
        recalc = preds_norm[0] * s + m

        assert np.isclose(
            preds_real[0], recalc, atol=1e-5
        ), "Denormalization logic mismatch."

    print(f"    Normalized Prediction [0]: {preds_norm[0]:.4f}")
    print(f"    Real Scale Prediction [0]: {preds_real[0]:.4f}")
    print("    Denormalization: OK")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
