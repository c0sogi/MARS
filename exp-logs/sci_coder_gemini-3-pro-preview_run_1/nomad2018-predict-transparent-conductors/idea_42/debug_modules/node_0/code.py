import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader

# Import library modules
import library.config as config
from library.data import process_data, get_scalers, CrystalDataset, collate_sparse
from library.model import REMSWDSModel
from library.train import train_model
from library.predict import generate_predictions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides default configuration for a quick demonstration run.
    """
    print("Setting up demo configuration...")

    # Define demo directories
    demo_working_dir = os.path.join(os.getcwd(), "working", "demo_execution")
    demo_submission_dir = os.path.join(os.getcwd(), "working", "demo_submission")
    demo_cache_dir = os.path.join(os.getcwd(), "working", "demo_cache")

    # Create directories
    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)

    # Patch config paths
    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = demo_submission_dir

    config.TRAIN_CACHE_PATH = os.path.join(demo_cache_dir, "train_data.npz")
    config.VAL_CACHE_PATH = os.path.join(demo_cache_dir, "val_data.npz")
    config.TEST_CACHE_PATH = os.path.join(demo_cache_dir, "test_data.npz")
    config.SCALERS_PATH = os.path.join(demo_working_dir, "scalers.npz")

    config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "best_model.pt")
    config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "demo_submission.csv")

    # Patch hyperparameters for speed
    config.EPOCHS = 2
    config.BATCH_SIZE = 16
    config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for demo

    print(f"  Working Dir: {config.WORKING_DIR}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print("-" * 40)


def demo_data_processing():
    """
    Demonstrates data processing and dataset creation.
    """
    print("\n[Demo] Data Processing & Loading")

    # Load metadata manually to pass to process_data
    train_df = pd.read_csv(config.TRAIN_CSV)

    # 1. Process Data
    # This extracts features from .xyz files
    print("  Processing training data subset...")
    atomic_feats, global_feats, targets, ids = process_data(
        train_df, config.TRAIN_CACHE_PATH, load_cached_data=False
    )

    # Verify outputs
    n_samples = len(ids)
    print(f"  Processed {n_samples} samples.")
    assert (
        n_samples == config.DEBUG_SAMPLE_SIZE
    ), f"Expected {config.DEBUG_SAMPLE_SIZE} samples, got {n_samples}"
    assert len(atomic_feats) == n_samples
    assert global_feats.shape == (n_samples, config.GLOBAL_FEATURES_DIM)
    assert targets.shape == (n_samples, 2)

    # 2. Get Scalers
    print("  Fitting scalers...")
    scaler_atomic, scaler_global = get_scalers(atomic_feats, global_feats)

    # 3. Create Dataset
    print("  Creating CrystalDataset...")
    dataset = CrystalDataset(
        atomic_feats,
        global_feats,
        targets,
        ids,
        scaler_atomic=scaler_atomic,
        scaler_global=scaler_global,
        mode="train",
    )

    # 4. Create DataLoader
    print("  Creating DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_sparse)

    # 5. Inspect a Batch
    batch = next(iter(loader))
    print("  Batch inspection:")
    print(
        f"    Atomic Features Shape: {batch['atomic_features'].shape} (N_atoms, {config.ATOM_FEATURES_DIM})"
    )
    print(f"    Batch Index Shape: {batch['batch_index'].shape} (N_atoms,)")
    print(
        f"    Global Features Shape: {batch['global_features'].shape} (Batch, {config.GLOBAL_FEATURES_DIM})"
    )
    print(f"    Targets Shape: {batch['targets'].shape} (Batch, 2)")

    assert batch["atomic_features"].shape[1] == config.ATOM_FEATURES_DIM
    assert batch["global_features"].shape[1] == config.GLOBAL_FEATURES_DIM
    assert batch["targets"].shape[1] == 2

    return batch


def demo_model_forward(batch):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n[Demo] Model Initialization & Forward Pass")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = REMSWDSModel(
        atom_features_dim=config.ATOM_FEATURES_DIM,
        global_features_dim=config.GLOBAL_FEATURES_DIM,
        hidden_dim=64,  # Reduced for demo
        atomic_layers=2,
        global_layers=2,
        fusion_layers=2,
    ).to(device)

    # Move batch to device
    af = batch["atomic_features"].to(device)
    bi = batch["batch_index"].to(device)
    gf = batch["global_features"].to(device)

    # Forward pass
    output = model(af, bi, gf)

    print(f"  Output Shape: {output.shape}")
    assert output.shape == (batch["targets"].shape[0], 2)
    print("  Forward pass successful.")


def demo_full_training():
    """
    Runs the full training pipeline using the library function.
    """
    print("\n[Demo] Full Training Pipeline")

    # This function handles loading, processing, training, and saving
    best_loss = train_model()

    print(f"  Training finished. Best Val Loss: {best_loss:.4f}")
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint was not saved."


def demo_inference():
    """
    Runs the inference pipeline using the library function.
    """
    print("\n[Demo] Inference Pipeline")

    # This function loads the model and generates predictions for the test set
    generate_predictions(load_cached_data=True)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated."

    # Verify submission format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"  Submission shape: {sub_df.shape}")
    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Since we used DEBUG_SAMPLE_SIZE on test set too (in process_data),
    # the submission should have that many rows.
    # Note: process_data applies debug slicing.
    assert (
        len(sub_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Expected {config.DEBUG_SAMPLE_SIZE} predictions, got {len(sub_df)}"


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Data Components
    batch_data = demo_data_processing()

    # 3. Model Components
    demo_model_forward(batch_data)

    # 4. Full Training Loop
    demo_full_training()

    # 5. Full Inference Loop
    demo_inference()

    print("\n[Demo] All demonstrations completed successfully.")
