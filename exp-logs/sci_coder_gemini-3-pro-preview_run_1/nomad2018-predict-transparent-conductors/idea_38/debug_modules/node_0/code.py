import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import logging

# Import library modules
import library.config as config
import library.utils as utils
import library.geometry_utils as geo_utils
import library.feature_engineering as feat_eng
import library.data_loader as dl
import library.architecture as arch
import library.training as training


def run_demo():
    print("=== Starting Demonstration of Library Modules ===")

    # 1. Setup & Configuration Override for Speed
    # We want to run on a small subset and store outputs in a separate directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch the WORKING_DIR in relevant modules to use our demo directory
    # This ensures we don't overwrite existing cache or models and keeps the demo clean
    config.WORKING_DIR = DEMO_DIR
    feat_eng.WORKING_DIR = DEMO_DIR
    training.WORKING_DIR = DEMO_DIR

    # Patch DEBUG_SAMPLE_SIZE to ensure fast execution of data loading
    dl.DEBUG_SAMPLE_SIZE = 50

    # Patch SUBMISSION_DIR
    DEMO_SUBMISSION_DIR = "./working/demo_submission"
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)
    training.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    utils.SUBMISSION_DIR = DEMO_SUBMISSION_DIR

    # Set seed
    utils.seed_everything(42)
    print("Configuration patched for demo execution.")

    # 2. Demonstrate utils.py
    print("\n--- Testing utils.py ---")
    logger = utils.get_logger(
        "demo_logger", log_file=os.path.join(DEMO_DIR, "demo.log")
    )
    logger.info("Logger initialized.")

    meter = utils.AverageMeter("TestMeter")
    meter.update(10.0)
    meter.update(20.0)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    print("AverageMeter verified.")

    y_true = torch.tensor([1.0, 2.0])
    y_pred = torch.tensor([1.1, 1.9])
    score = utils.rmsle(y_true, y_pred)
    print(f"RMSLE score: {score.item():.4f}")
    assert not torch.isnan(score), "RMSLE returned NaN"

    # 3. Demonstrate geometry_utils.py
    print("\n--- Testing geometry_utils.py ---")
    # Use a real file from the dataset
    sample_rel_path = "train/1/geometry.xyz"
    atoms = geo_utils.load_atoms(sample_rel_path)
    print(f"Loaded atoms from {sample_rel_path}: {len(atoms)} atoms")

    distances = geo_utils.get_pbc_distances(atoms)
    assert distances.shape == (len(atoms), len(atoms)), "Distance matrix shape mismatch"

    centered_pos = geo_utils.get_centered_positions(atoms)
    assert centered_pos.shape == (len(atoms), 3), "Centered positions shape mismatch"
    # Centroid should be close to 0
    centroid = np.mean(centered_pos, axis=0)
    assert np.allclose(centroid, 0, atol=1e-5), "Centering failed"
    print("Geometry utils verified.")

    # 4. Demonstrate feature_engineering.py
    print("\n--- Testing feature_engineering.py ---")
    # Create a dummy row similar to metadata
    dummy_row = {
        "lattice_vector_1_ang": 5.0,
        "lattice_vector_2_ang": 5.0,
        "lattice_vector_3_ang": 10.0,
        "lattice_angle_alpha_degree": 90.0,
        "lattice_angle_beta_degree": 90.0,
        "lattice_angle_gamma_degree": 90.0,
        "number_of_total_atoms": len(atoms),
        "percent_atom_al": 0.25,
        "percent_atom_ga": 0.25,
        "percent_atom_in": 0.50,
    }

    struct_stats = feat_eng.compute_structural_stats(atoms, distances)
    print(f"Structural stats shape: {struct_stats.shape}")
    assert struct_stats.shape[0] == len(
        config.PAIR_TYPES
    ), "Structural stats dimension mismatch"

    atom_feats = feat_eng.extract_atomic_features(atoms, distances)
    print(f"Atomic features shape: {atom_feats.shape}")
    assert (
        atom_feats.shape[1] == config.ATOMIC_FEATURE_DIM
    ), "Atomic feature dim mismatch"

    global_feats = feat_eng.extract_global_features(dummy_row, struct_stats)
    print(f"Global features shape: {global_feats.shape}")
    assert (
        global_feats.shape[0] == config.GLOBAL_FEATURE_DIM
    ), "Global feature dim mismatch"
    print("Feature engineering functions verified.")

    # 5. Demonstrate data_loader.py
    print("\n--- Testing data_loader.py ---")
    # This will trigger prepare_data which computes features for the subset (DEBUG_SAMPLE_SIZE=50)
    # and caches them in DEMO_DIR
    train_loader, val_loader, test_loader = dl.get_dataloaders(
        load_cached_data=False, batch_size=4
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())

    atomic = batch["atomic"]
    mask = batch["atomic_mask"]
    glob = batch["global"]
    targets = batch["targets"]
    ids = batch["ids"]

    print(f"Atomic batch shape: {atomic.shape}")  # (B, N_max, D_atom)
    print(f"Global batch shape: {glob.shape}")  # (B, D_global)
    print(f"Targets batch shape: {targets.shape}")  # (B, 2)

    assert atomic.shape[0] == 4, "Batch size mismatch"
    assert (
        atomic.shape[2] == config.ATOMIC_FEATURE_DIM
    ), "Atomic feature dim in loader mismatch"
    assert (
        glob.shape[1] == config.GLOBAL_FEATURE_DIM
    ), "Global feature dim in loader mismatch"
    assert targets.shape[1] == 2, "Target dim mismatch"
    print("Data loader verified.")

    # 6. Demonstrate architecture.py
    print("\n--- Testing architecture.py ---")
    model = arch.SSAWDSModel(
        atomic_input_dim=config.ATOMIC_FEATURE_DIM,
        global_input_dim=config.GLOBAL_FEATURE_DIM,
        atomic_hidden_dim=32,  # Reduced for demo speed
        global_hidden_dim=32,
        fusion_hidden_dim=32,
    )

    # Run forward pass with the batch from data loader
    # Move to CPU for this test
    model.eval()
    with torch.no_grad():
        outputs = model(atomic, mask, glob)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (4, 2), "Model output shape mismatch"
    print("Model architecture verified.")

    # 7. Demonstrate training.py (End-to-End)
    print("\n--- Testing training.py (End-to-End) ---")
    # We will run a very short training cycle
    # run_training will re-initialize the model with default config sizes, which is fine

    # Ensure we use the demo directory
    # Note: run_training uses get_dataloaders internally, which uses the patched DEBUG_SAMPLE_SIZE

    training.run_training(
        epochs=1,
        batch_size=8,
        load_cached_data=True,  # Use the data we just cached in step 5
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    # Verify submission file creation
    submission_path = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        print(f"Submission file created at {submission_path}")
        sub_df = pd.read_csv(submission_path)
        print(f"Submission shape: {sub_df.shape}")
        # In debug mode, test set is also subsampled to 50
        assert (
            len(sub_df) == 50
        ), f"Expected 50 predictions in debug mode, got {len(sub_df)}"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
