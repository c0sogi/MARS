import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import importlib

# Force reload of library modules to pick up changes
import library.config
import library.utils
import library.data
import library.model
import library.train

importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.data)
importlib.reload(library.model)
importlib.reload(library.train)

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    compute_pbc_distances,
    compute_inertia_eigenvalues,
    rmsle,
)
from library.data import (
    preprocess_dataset,
    fit_and_save_scalers,
    get_dataloaders,
    MaterialDataset,
    collate_fn,
)
from library.model import ACC_WDS
from library.train import run_training, train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def verify_utils():
    """
    Verifies the correctness of utility functions.
    """
    print("Verifying utility functions...")

    # 1. Verify PBC Distances
    # Simple cubic lattice 10x10x10
    lattice = np.diag([10.0, 10.0, 10.0])
    # Point A at 1.0, Point B at 9.0 along x-axis
    # Distance should be 2.0 (wrapping around boundary), not 8.0
    coords = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])

    dists, vecs = compute_pbc_distances(coords, lattice)

    # dists[0, 1] should be 2.0
    assert np.isclose(
        dists[0, 1], 2.0
    ), f"PBC Distance calc failed. Expected 2.0, got {dists[0, 1]}"
    # Vector from 0 to 1 should be [2.0, 0, 0] or [-2.0, 0, 0] depending on direction,
    # actually MIC: 1.0 - 9.0 = -8.0. In fractional: 0.1 - 0.9 = -0.8. MIC -> 0.2. 0.2 * 10 = 2.0.
    # So vector 0->1 is +2.0 along x.
    # Wait, diff_frac = frac[i] - frac[j].
    # i=0 (1.0), j=1 (9.0). frac 0.1, 0.9. diff = -0.8. round(-0.8) = -1.0.
    # diff_frac_mic = -0.8 - (-1.0) = 0.2.
    # diff_vec = 0.2 * 10 = 2.0.
    assert np.isclose(
        vecs[0, 1, 0], 2.0
    ), f"PBC Vector calc failed. Expected 2.0, got {vecs[0, 1, 0]}"

    # 2. Verify Inertia Eigenvalues
    # Create a dummy set of neighbor vectors: 3 points along axes
    neighbor_vecs = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    eigvals = compute_inertia_eigenvalues(neighbor_vecs)
    # Covariance of these points.
    # This checks that the function runs and returns 3 sorted values
    assert eigvals.shape == (3,), "Eigenvalues shape mismatch"
    assert eigvals[0] <= eigvals[1] <= eigvals[2], "Eigenvalues not sorted"

    print("Utility functions verified.")


def setup_demo_environment():
    """
    Sets up a temporary environment with a subset of data for speed.
    """
    print("\nSetting up demo environment...")

    # Define demo paths
    demo_dir = os.path.join("working", "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    # Create cache dir
    cache_dir = os.path.join("working", "demo_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Override Config paths to point to demo locations
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_DATA_CACHE = os.path.join(cache_dir, "train_data.npz")
    Config.VAL_DATA_CACHE = os.path.join(cache_dir, "val_data.npz")
    Config.TEST_DATA_CACHE = os.path.join(cache_dir, "test_data.npz")
    Config.SCALERS_CACHE = os.path.join(demo_dir, "scalers.npz")
    Config.MODEL_CHECKPOINT = os.path.join(demo_dir, "demo_model.pt")

    # Create subset metadata files
    # Read original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample small subset (e.g., 20 samples each)
    subset_size = 20
    demo_train = orig_train.head(subset_size)
    demo_val = orig_val.head(subset_size)
    demo_test = orig_test.head(subset_size)

    # Save subset metadata
    demo_train_path = os.path.join(demo_dir, "train.csv")
    demo_val_path = os.path.join(demo_dir, "val.csv")
    demo_test_path = os.path.join(demo_dir, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Override Config metadata paths
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path

    # Adjust Hyperparameters for Speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print("Demo environment configured.")


def demonstrate_data_pipeline():
    """
    Demonstrates data loading and processing.
    """
    print("\nDemonstrating Data Pipeline...")

    # 1. Preprocess (this will use the subset metadata created above)
    # We set load_cached_data=False to force processing
    train_data = preprocess_dataset(
        Config.TRAIN_META_PATH, Config.TRAIN_DATA_CACHE, load_cached_data=False
    )

    # Verify data structure
    assert len(train_data["ids"]) == 20, "Incorrect number of training samples"
    assert (
        train_data["atomic_features"][0].shape[1] == Config.ATOMIC_FEATURE_DIM
    ), "Incorrect atomic feature dim"
    assert (
        train_data["global_features"][0].shape[0] == Config.GLOBAL_FEATURE_DIM
    ), "Incorrect global feature dim"

    # 2. Scalers
    print("Fitting scalers...")
    fit_and_save_scalers(train_data, Config.SCALERS_CACHE)
    assert os.path.exists(Config.SCALERS_CACHE), "Scalers file not created"

    # 3. Get DataLoaders
    # This will process val and test as well
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Check batch structure
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Atomic Features Shape:", batch["atomic_features"].shape)  # (B, N_max, Feat)
    print("Global Features Shape:", batch["global_features"].shape)  # (B, Feat)
    print("Mask Shape:", batch["mask"].shape)  # (B, N_max)
    print("Targets Shape:", batch["target"].shape)  # (B, 2)

    assert batch["atomic_features"].ndim == 3
    assert batch["global_features"].ndim == 2
    assert batch["target"].shape[1] == 2

    return train_loader, val_loader, test_loader


def demonstrate_model_training(train_loader, val_loader):
    """
    Demonstrates model instantiation and training loop.
    """
    print("\nDemonstrating Model Training...")

    device = torch.device(
        "cpu"
    )  # Use CPU for demo to avoid CUDA initialization overhead if any
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Instantiate Model
    model = ACC_WDS().to(device)
    print("Model instantiated.")

    # Check forward pass with a batch
    batch = next(iter(train_loader))
    af = batch["atomic_features"].to(device)
    gf = batch["global_features"].to(device)
    mask = batch["mask"].to(device)

    output = model(af, gf, mask)
    assert output.shape == (
        batch["ids"].shape[0],
        2,
    ), f"Output shape mismatch: {output.shape}"
    print("Forward pass successful.")

    # Run Training
    # We use the run_training function from library.train but we need to ensure it uses our modified Config
    # Since run_training internally calls get_dataloaders and re-initializes everything based on Config,
    # and we have modified Config class attributes in memory, it should work fine.

    print("Running training loop (2 epochs)...")
    run_training(load_cached_data=True)

    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), "Model checkpoint not found after training"
    print("Training demonstration complete.")


def demonstrate_inference(test_loader):
    """
    Demonstrates loading a model and making predictions.
    """
    print("\nDemonstrating Inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = ACC_WDS().to(device)
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            af = batch["atomic_features"].to(device)
            gf = batch["global_features"].to(device)
            mask = batch["mask"].to(device)

            out = model(af, gf, mask)
            # Inverse transform: expm1
            preds = torch.expm1(out).cpu().numpy()

            predictions.append(preds)
            ids.extend(batch["ids"].numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Create submission dataframe
    sub_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID
    sub_df.sort_values("id", inplace=True)

    # Save
    sub_dir = os.path.join("working", "demo_submission")
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "demo_submission.csv")
    sub_df.to_csv(sub_path, index=False)

    print(f"Inference complete. Submission saved to {sub_path}")
    print("Head of submission:")
    print(sub_df.head())


def main():
    # 1. Set Seed
    set_seed(42)

    # 2. Verify Utils
    verify_utils()

    # 3. Setup Environment (Subset Data)
    setup_demo_environment()

    # 4. Data Pipeline
    train_loader, val_loader, test_loader = demonstrate_data_pipeline()

    # 5. Model & Training
    demonstrate_model_training(train_loader, val_loader)

    # 6. Inference
    demonstrate_inference(test_loader)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
