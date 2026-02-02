import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.geometry_utils import (
    parse_xyz,
    get_pbc_distances,
    compute_chemical_densities,
)
from library.data_processing import (
    FeatureExtractor,
    PreprocessPipeline,
    process_dataset,
)
from library.dataset import MaterialsDataset, collate_fn
from library.model import GDCC_WDS
from library.train import Trainer, train_model
from library.inference import generate_submission, run_inference


def run_demo():
    print("=" * 80)
    print("Running Library Demonstration and Verification")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 0. Setup and Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("\n[Step 0] Configuring environment for demo...")

    # Override Config to use a specific demo directory and reduce compute load
    DEMO_DIR = "./working/demo_execution"
    Config.WORKING_DIR = DEMO_DIR
    # Cite debug_lesson_7: Manually Update Derived Static Configuration Variables After Runtime Changes
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pt")

    Config.NUM_EPOCHS = 2  # Run only 2 epochs for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure demo directory exists
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    Config.make_dirs()

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 1. Verify Geometry Utils
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying Geometry Utils...")

    # Pick a sample file
    sample_id = 1
    sample_rel_path = f"train/{sample_id}/geometry.xyz"
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    if not os.path.exists(sample_full_path):
        raise FileNotFoundError(f"Sample file not found: {sample_full_path}")

    # Test parse_xyz
    lattice, coords, types = parse_xyz(sample_full_path)
    print(f"  Parsed {sample_rel_path}: {len(coords)} atoms.")

    assert lattice.shape == (3, 3), f"Lattice shape mismatch: {lattice.shape}"
    assert coords.shape[1] == 3, f"Coords shape mismatch: {coords.shape}"
    assert len(types) == len(coords), "Mismatch between types and coords length"

    # Test get_pbc_distances
    dists = get_pbc_distances(coords, lattice)
    print(f"  Computed PBC distances matrix: {dists.shape}")
    assert dists.shape == (len(coords), len(coords)), "Distance matrix shape mismatch"
    assert np.allclose(dists.diagonal(), 0), "Diagonal distances should be 0"

    # Test compute_chemical_densities
    densities = compute_chemical_densities(
        coords,
        types,
        lattice,
        Config.ATOM_TYPES,
        Config.DENSITY_GAMMA,
        Config.DENSITY_CUTOFF,
    )
    print(f"  Computed chemical densities: {densities.shape}")
    assert densities.shape == (
        len(coords),
        len(Config.ATOM_TYPES),
    ), "Density shape mismatch"

    print("  Geometry utils verified.")

    # -------------------------------------------------------------------------
    # 2. Verify Data Processing (Feature Extraction)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Feature Extraction...")

    extractor = FeatureExtractor()
    atomic_feats, global_feats = extractor.process_sample(sample_rel_path)

    print(f"  Atomic features shape: {atomic_feats.shape}")
    print(f"  Global features shape: {global_feats.shape}")

    assert (
        atomic_feats.shape[1] == Config.ATOMIC_INPUT_DIM
    ), f"Atomic dim mismatch. Expected {Config.ATOMIC_INPUT_DIM}, got {atomic_feats.shape[1]}"
    assert (
        global_feats.shape[0] == Config.GLOBAL_INPUT_DIM
    ), f"Global dim mismatch. Expected {Config.GLOBAL_INPUT_DIM}, got {global_feats.shape[0]}"

    # Test Pipeline Scaling
    pipeline = PreprocessPipeline()
    # Create dummy batch
    dummy_atomic = np.random.randn(100, Config.ATOMIC_INPUT_DIM)
    dummy_global = np.random.randn(10, Config.GLOBAL_INPUT_DIM)

    pipeline.fit(dummy_atomic, dummy_global)
    scaled_a, scaled_g = pipeline.transform(dummy_atomic, dummy_global)

    assert np.allclose(
        scaled_a.mean(axis=0), 0, atol=1e-5
    ), "Atomic scaling mean failed"
    assert np.allclose(
        scaled_g.mean(axis=0), 0, atol=1e-5
    ), "Global scaling mean failed"

    print("  Feature extraction and scaling verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Dataset and DataLoader...")

    # Use a small subset
    MAX_SAMPLES = 10

    # Initialize dataset (this will trigger processing and caching)
    train_dataset = MaterialsDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        mode="train",
        max_samples=MAX_SAMPLES,
        load_cached_data=False,  # Force re-process for demo
    )

    print(f"  Dataset initialized with {len(train_dataset)} samples.")
    assert (
        len(train_dataset) == MAX_SAMPLES
    ), f"Dataset length mismatch: {len(train_dataset)}"

    # Check single item
    item = train_dataset[0]
    print("  Sample item keys:", item.keys())
    assert "atomic_features" in item
    assert "global_features" in item
    assert "targets" in item
    assert "id" in item

    # Check DataLoader collation
    loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn
    )
    batch = next(iter(loader))

    print("  Batch keys:", batch.keys())
    print(f"  Batch atomic features: {batch['atomic_features'].shape}")
    print(f"  Batch global features: {batch['global_features'].shape}")
    print(f"  Batch targets: {batch['targets'].shape}")

    assert batch["global_features"].shape[0] == Config.BATCH_SIZE
    assert batch["targets"].shape[0] == Config.BATCH_SIZE
    assert batch["targets"].shape[1] == Config.OUTPUT_DIM

    print("  Dataset and DataLoader verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    model = GDCC_WDS()
    model.to(Config.DEVICE)

    # Prepare inputs from batch
    af = batch["atomic_features"].to(Config.DEVICE)
    gf = batch["global_features"].to(Config.DEVICE)
    bi = batch["batch_indices"].to(Config.DEVICE)
    num_graphs = len(batch["ids"])

    # Forward pass
    output = model(af, gf, bi, num_graphs)

    print(f"  Model output shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.OUTPUT_DIM,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.OUTPUT_DIM)}, got {output.shape}"

    print("  Model architecture verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying Training Loop...")

    # We will run the actual train_model function but with limited samples/epochs
    # This function handles dataset creation, model init, and training
    train_model(
        max_samples=20,  # Small subset for speed
        epochs=2,  # Minimal epochs
        batch_size=4,
        load_cached_data=False,  # Force fresh processing
    )

    # Check if model file was created
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"  Model saved successfully at {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Model file was not created after training.")

    print("  Training loop verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Inference and Submission...")

    submission_path = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Run generation on a subset of test data
    generate_submission(
        model_path=Config.BEST_MODEL_PATH,
        output_path=submission_path,
        max_samples=10,
        load_cached_data=False,
    )

    if os.path.exists(submission_path):
        print(f"  Submission file created at {submission_path}")
        df_sub = pd.read_csv(submission_path)
        print("  Submission head:")
        print(df_sub.head())

        assert len(df_sub) == 10, "Submission length mismatch"
        assert list(df_sub.columns) == [
            "id",
            "formation_energy_ev_natom",
            "bandgap_energy_ev",
        ], "Submission columns mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("  Inference verified.")

    print("\n" + "=" * 80)
    print("All checks passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
