import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import library modules
from library.config import Config
from library.geometry_utils import parse_xyz, compute_atomic_features
from library.data_loader import get_loaders, compute_global_features
from library.model import DC3_WDS
from library.trainer import run_training
from library.inference import run_inference

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo_geometry_utils():
    print("\n--- Demo: Geometry Utils ---")

    # Pick a sample file (ID 1 from train set)
    sample_id = 1
    rel_path = f"train/{sample_id}/geometry.xyz"
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        print(f"Sample file {full_path} not found. Skipping geometry demo.")
        return

    # 1. Parse XYZ
    lattice, atom_types, coords = parse_xyz(full_path)
    print(f"Parsed {rel_path}:")
    print(f"  Lattice shape: {lattice.shape}")
    print(f"  Num atoms: {len(atom_types)}")
    print(f"  Coords shape: {coords.shape}")

    assert lattice.shape == (3, 3), "Lattice shape mismatch"
    assert len(atom_types) == coords.shape[0], "Atom types/coords count mismatch"

    # 2. Compute Atomic Features
    # Note: Config.ATOM_MAP and Config.K_NEIGHBORS are used inside
    features = compute_atomic_features(
        atom_types, coords, lattice, k_neighbors=Config.K_NEIGHBORS
    )
    print(f"  Atomic Features shape: {features.shape}")

    expected_dim = Config.ATOMIC_FEATURE_DIM
    assert features.shape == (
        len(atom_types),
        expected_dim,
    ), f"Expected feature shape ({len(atom_types)}, {expected_dim}), got {features.shape}"

    # 3. Compute Global Features (Mocking a dataframe row)
    # Mock row based on typical values
    mock_row = {
        "percent_atom_al": 0.25,
        "percent_atom_ga": 0.25,
        "percent_atom_in": 0.25,
    }
    global_feats = compute_global_features(lattice, atom_types, mock_row)
    print(f"  Global Features shape: {global_feats.shape}")

    expected_global_dim = Config.GLOBAL_FEATURE_DIM
    assert global_feats.shape == (
        expected_global_dim,
    ), f"Expected global feature shape ({expected_global_dim},), got {global_feats.shape}"

    print("Geometry utils verification passed.")


def demo_data_loader():
    print("\n--- Demo: Data Loader ---")

    # get_loaders will use the patched Config to load only a subset
    # load_cached_data=False forces reprocessing for this demo
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))

    atomic_feats = batch["atomic_feats"]
    global_feats = batch["global_feats"]
    mask = batch["mask"]
    targets = batch["target"]
    ids = batch["id"]

    print("Batch shapes:")
    print(f"  Atomic Feats: {atomic_feats.shape}")  # (B, N_max, 13)
    print(f"  Global Feats: {global_feats.shape}")  # (B, 12)
    print(f"  Mask: {mask.shape}")  # (B, N_max)
    print(f"  Targets: {targets.shape}")  # (B, 2)
    print(f"  IDs: {ids.shape}")  # (B,)

    assert atomic_feats.ndim == 3
    assert atomic_feats.shape[2] == Config.ATOMIC_FEATURE_DIM
    assert global_feats.shape[1] == Config.GLOBAL_FEATURE_DIM
    assert targets.shape[1] == Config.NUM_TARGETS

    print("Data loader verification passed.")
    return batch


def demo_model(batch):
    print("\n--- Demo: Model Forward Pass ---")

    device = torch.device("cpu")  # Use CPU for simple demo check
    model = DC3_WDS().to(device)

    # Move batch to device
    atomic_feats = batch["atomic_feats"].to(device)
    global_feats = batch["global_feats"].to(device)
    mask = batch["mask"].to(device)

    # Forward
    output = model(atomic_feats, global_feats, mask)

    print(f"Model Output shape: {output.shape}")

    assert output.shape == (
        atomic_feats.shape[0],
        Config.NUM_TARGETS,
    ), f"Output shape mismatch. Expected {(atomic_feats.shape[0], Config.NUM_TARGETS)}, got {output.shape}"

    print("Model forward pass verification passed.")


def demo_training_pipeline():
    print("\n--- Demo: Full Training Pipeline ---")

    # run_training handles loading data, initializing model, training loop, and submission generation
    # It uses the Config settings we patched.
    # We set load_cached_data=True to reuse the data processed in demo_data_loader
    run_training(load_cached_data=True)

    # Check artifacts
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint found at: {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created.")

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file found at: {Config.SUBMISSION_PATH}")
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission head:")
        print(df.head())
        assert len(df) > 0, "Submission file is empty"
        assert list(df.columns) == ["id"] + Config.TARGET_COLS
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("Training pipeline verification passed.")


def demo_inference_pipeline():
    print("\n--- Demo: Inference Pipeline ---")

    # run_inference loads the model from Config.MODEL_PATH and generates predictions on test set
    run_inference(load_cached_data=True)

    # Verify the submission file is updated (timestamp check could be done, but existence is sufficient here)
    if os.path.exists(Config.SUBMISSION_PATH):
        print("Inference completed successfully.")
    else:
        raise FileNotFoundError("Inference failed to produce submission file.")


def main():
    # 1. Setup
    set_seed(42)

    # 2. Patch Configuration for Demo Speed
    print("Configuring for demo...")
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pt")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples for speed

    # Reduce training parameters
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Re-initialize directories
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 3. Run Demos
    try:
        demo_geometry_utils()

        # Get a batch for model testing
        batch = demo_data_loader()

        demo_model(batch)

        demo_training_pipeline()

        demo_inference_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
