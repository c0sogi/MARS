import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library import utils, geometry, data, model, engine


def main():
    print("Initializing AMSA-DS Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Create a separate working directory for this demo to avoid interference
    # with any ongoing full-scale runs.
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir)

    # Override Config parameters for a quick demonstration
    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_DIR = os.path.join(demo_working_dir, "cache")
    Config.EXECUTION_DIR = os.path.join(demo_working_dir, "execution")
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")

    # Re-run setup to create these new directories
    Config.setup()

    # Set Debug mode to process only a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SIZE = 20  # Only process 20 crystals
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small script

    # Use CPU for demo stability/simplicity unless GPU is strictly desired
    # The library code uses Config.DEVICE, so we can override it here if needed.
    # We will stick to the library's detection logic but print it.
    print(f"Device: {Config.DEVICE}")
    print(
        f"Configured: DEBUG={Config.DEBUG}, EPOCHS={Config.NUM_EPOCHS}, BATCH={Config.BATCH_SIZE}"
    )

    # -------------------------------------------------------------------------
    # 2. Utils Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Utils ---")
    utils.set_seed(42)

    # Test Transformations
    val = np.array([1.0, 10.0])
    log_val = utils.log1p_transform(val)
    exp_val = utils.expm1_transform(log_val)
    assert np.allclose(val, exp_val), "Log1p/Expm1 transform mismatch"
    print("Transforms verified.")

    # Test SelectiveScaler
    # Create dummy data: Col 0 is constant/categorical-like, Col 1 is continuous
    X_dummy = np.array([[1.0, 100.0], [1.0, 200.0], [1.0, 300.0]])
    scaler = utils.SelectiveScaler(cols_to_scale=[1])
    scaler.fit(X_dummy)
    X_scaled = scaler.transform(X_dummy)

    # Check that Col 0 is untouched and Col 1 is standardized (mean=0, std=1)
    assert X_scaled[0, 0] == 1.0, "Column 0 should not be scaled"
    assert abs(np.mean(X_scaled[:, 1])) < 1e-6, "Column 1 should be centered"
    assert abs(np.std(X_scaled[:, 1]) - 1.0) < 1e-6, "Column 1 should be scaled"
    print("SelectiveScaler verified.")

    # -------------------------------------------------------------------------
    # 3. Geometry Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Geometry ---")
    # Pick a sample file from metadata to test parsing
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    sample_rel_path = train_df.iloc[0]["file_path"]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    # Test Parsing
    lattice, types, coords = geometry.parse_xyz(sample_full_path)
    print(f"Parsed {sample_rel_path}: {len(types)} atoms.")
    assert len(lattice) == 3, "Lattice should have 3 vectors"
    assert len(coords) == len(types), "Mismatch between atoms and coordinates"

    # Test Neighbor Finding
    dists, indices = geometry.get_pbc_neighbors(coords, lattice, k_max=Config.K_FAR)
    assert dists.shape == (len(types), Config.K_FAR), "Neighbor distance shape mismatch"
    assert indices.shape == (
        len(types),
        Config.K_FAR,
    ), "Neighbor indices shape mismatch"
    print("Neighbor finding verified.")

    # Test Feature Computation
    atomic_feats = geometry.compute_atomic_features(coords, types, dists, indices)
    global_feats = geometry.compute_global_features(lattice, types)

    assert (
        atomic_feats.shape[1] == Config.ATOMIC_FEATURE_DIM
    ), f"Atomic dim mismatch: got {atomic_feats.shape[1]}, expected {Config.ATOMIC_FEATURE_DIM}"
    assert (
        global_feats.shape[0] == Config.GLOBAL_FEATURE_DIM
    ), f"Global dim mismatch: got {global_feats.shape[0]}, expected {Config.GLOBAL_FEATURE_DIM}"

    print(f"Atomic features shape: {atomic_feats.shape}")
    print(f"Global features shape: {global_feats.shape}")

    # -------------------------------------------------------------------------
    # 4. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loading ---")
    # This calls process_dataset internally. Since DEBUG=True, it processes a small subset.
    # load_cached_data=False ensures we run the processing logic.
    train_loader, val_loader = data.get_train_val_loaders(load_cached_data=False)

    # Fetch a single batch to verify collation
    batch = next(iter(train_loader))
    b_atomic, b_global, b_indices, b_targets, b_ids = batch

    print(f"Batch Atomic Tensor: {b_atomic.shape}")
    print(f"Batch Global Tensor: {b_global.shape}")
    print(f"Batch Indices Tensor: {b_indices.shape}")
    print(f"Batch Targets Tensor: {b_targets.shape}")

    # Verification
    assert b_atomic.dim() == 2
    assert b_global.dim() == 2
    assert b_indices.dim() == 1
    assert b_targets.dim() == 2
    assert b_atomic.shape[1] == Config.ATOMIC_FEATURE_DIM
    assert b_global.shape[1] == Config.GLOBAL_FEATURE_DIM
    # Check that batch indices correspond to valid batch items (0 to BATCH_SIZE-1)
    assert b_indices.max() < Config.BATCH_SIZE
    print("DataLoaders verified.")

    # -------------------------------------------------------------------------
    # 5. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Model ---")
    net = model.AMSA_DS()
    device = torch.device(Config.DEVICE)
    net.to(device)

    # Move batch to device
    b_atomic = b_atomic.to(device)
    b_global = b_global.to(device)
    b_indices = b_indices.to(device)

    # Forward pass
    out = net(b_atomic, b_global, b_indices)
    print(f"Model Output Shape: {out.shape}")

    # Output should be (Batch_Size, 2)
    assert out.shape == (b_global.shape[0], 2)
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 6. Engine Demonstration (Training & Inference)
    # -------------------------------------------------------------------------
    print("\n--- Testing Engine (Training & Inference) ---")

    # Run full training loop (shortened by Config overrides)
    # We can load cached data this time since we just generated it in step 4
    best_loss = engine.run_training(load_cached_data=True)
    print(f"Training finished with best validation loss: {best_loss:.4f}")

    # Run submission generation
    # get_test_loader will process the test set (also subsetted by DEBUG)
    engine.generate_submission(load_cached_data=False)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated with {len(df_sub)} rows.")
        print(df_sub.head())

        # Validate submission format
        assert "id" in df_sub.columns
        assert "formation_energy_ev_natom" in df_sub.columns
        assert "bandgap_energy_ev" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
