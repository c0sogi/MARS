import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library modules
from library import config
from library import geometry
from library import features
from library import dataset
from library import model
from library import trainer


def main():
    print("Initializing demonstration...")

    # =========================================================================
    # 1. Configuration Override for Speed
    # =========================================================================
    print("\n[1] Overriding Configuration for Demo Speed")
    # Use a separate working directory for this demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override config constants
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.TRAIN_CACHE = os.path.join(DEMO_WORKING_DIR, "train_data.npz")
    config.VAL_CACHE = os.path.join(DEMO_WORKING_DIR, "val_data.npz")
    config.TEST_CACHE = os.path.join(DEMO_WORKING_DIR, "test_data.npz")
    config.SCALERS_CACHE = os.path.join(DEMO_WORKING_DIR, "scalers.npz")
    config.MODEL_CHECKPOINT = os.path.join(DEMO_WORKING_DIR, "demo_model.pt")

    # Set training hyperparameters to minimal values
    config.EPOCHS = 1
    config.BATCH_SIZE = 16
    config.PATIENCE = 1

    # Set submission path
    config.SUBMISSION_DIR = "./working/demo_submission"
    config.SUBMISSION_FILE = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")

    # =========================================================================
    # 2. Geometry Module Demonstration
    # =========================================================================
    print("\n[2] Testing Geometry Module")

    # Pick a sample file (e.g., from test set as we know the structure)
    sample_xyz_path = os.path.join(config.INPUT_DIR, "test/1/geometry.xyz")
    if os.path.exists(sample_xyz_path):
        lattice, atom_types, coords = geometry.read_xyz(sample_xyz_path)
        print(f"Read XYZ: {sample_xyz_path}")
        print(f"Lattice shape: {lattice.shape}")
        print(f"Number of atoms: {len(atom_types)}")
        print(f"Coordinates shape: {coords.shape}")

        assert lattice.shape == (3, 3), "Lattice matrix should be 3x3"
        assert (
            len(atom_types) == coords.shape[0]
        ), "Mismatch between atom types and coordinates"

        # Test conversion to fractional
        frac_coords = geometry.cartesian_to_fractional(coords, lattice)
        assert frac_coords.shape == coords.shape, "Fractional coords shape mismatch"

        # Test PBC distances
        dist_matrix = geometry.get_pbc_distances(frac_coords, lattice)
        assert dist_matrix.shape == (
            len(atom_types),
            len(atom_types),
        ), "Distance matrix shape mismatch"

        # Test Potential
        potential, nn_dist = geometry.compute_local_potential(dist_matrix)
        assert potential.shape == (len(atom_types),), "Potential shape mismatch"
        assert nn_dist.shape == (len(atom_types),), "NN dist shape mismatch"
        print("Geometry functions verified.")
    else:
        print(
            f"Warning: Sample file {sample_xyz_path} not found. Skipping geometry file read test."
        )

    # =========================================================================
    # 3. Feature Extraction Demonstration
    # =========================================================================
    print("\n[3] Testing Feature Extraction")

    # Load a small subset of metadata to test extraction without processing everything
    full_train_df = pd.read_csv(config.TRAIN_CSV)
    subset_df = full_train_df.head(10).copy()  # Process only 10 samples

    extractor = features.MaterialFeatureExtractor()
    scaler = features.DataScaler()

    print("Processing subset of training data...")
    # We force load_cached_data=False to verify the extraction logic
    data_dict = extractor.process_data(
        subset_df, split_name="train", load_cached_data=False, scaler=scaler
    )

    atomic_feats = data_dict["atomic_features"]
    global_feats = data_dict["global_features"]
    targets = data_dict["targets"]
    ids = data_dict["ids"]

    print(f"Processed {len(ids)} samples.")
    print(f"Global features shape: {global_feats.shape}")
    print(f"Targets shape: {targets.shape}")

    assert len(atomic_feats) == len(ids), "Atomic features list length mismatch"
    assert (
        global_feats.shape[1] == config.GLOBAL_INPUT_DIM
    ), f"Global dim mismatch. Expected {config.GLOBAL_INPUT_DIM}, got {global_feats.shape[1]}"
    assert scaler.fitted, "Scaler should be fitted after processing train data"

    # Verify Scaler saving/loading
    scaler.save(config.SCALERS_CACHE)
    loaded_scaler = features.DataScaler()
    loaded_scaler.load(config.SCALERS_CACHE)
    assert loaded_scaler.fitted, "Loaded scaler should be fitted"
    print("Feature extraction and scaling verified.")

    # =========================================================================
    # 4. Dataset and DataLoader Demonstration
    # =========================================================================
    print("\n[4] Testing Dataset and DataLoader")

    ds = dataset.MaterialsDataset(atomic_feats, global_feats, targets, ids)
    item0 = ds[0]
    print(f"Sample 0 ID: {item0['id']}")
    print(f"Sample 0 Atomic Feat Shape: {item0['atomic_features'].shape}")

    # Create a dataloader with the subset
    # Using collate_batch from dataset module
    dl = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=dataset.collate_batch)

    batch = next(iter(dl))
    print("Batch keys:", batch.keys())
    print(f"Batch Atomic Features: {batch['atomic_features'].shape}")
    print(f"Batch Mask: {batch['mask'].shape}")
    print(f"Batch Targets: {batch['targets'].shape}")

    # Check masking logic: padded entries should have mask=0
    # The length of atomic features in dim 1 is max_atoms in batch
    # We check if mask sums match actual atom counts (which we can infer from non-zero rows if padding is 0,
    # but collate_batch explicitly creates mask based on lengths)
    assert batch["atomic_features"].ndim == 3
    assert batch["mask"].ndim == 2
    print("DataLoader and Collation verified.")

    # =========================================================================
    # 5. Model Demonstration
    # =========================================================================
    print("\n[5] Testing Model Forward Pass")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.HCPDS().to(device)

    # Move batch to device
    b_atomic = batch["atomic_features"].to(device)
    b_global = batch["global_features"].to(device)
    b_mask = batch["mask"].to(device)

    # Forward
    output = net(b_atomic, b_global, b_mask)
    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (4, 2), "Output shape mismatch. Expected (Batch, 2)"
    print("Model forward pass verified.")

    # =========================================================================
    # 6. Full Training Loop Demonstration
    # =========================================================================
    print("\n[6] Running Full Training Loop (1 Epoch)")

    # We will use the trainer.run_training function.
    # Note: This function loads data internally using get_dataloader.
    # get_dataloader reads config.TRAIN_CSV.
    # Since we want to run fast, we rely on the config overrides we set at the start.
    # However, get_dataloader will process the FULL dataset defined in metadata/train.csv
    # if we don't provide a cache. Processing 1700 files is reasonably fast (~10-20s).
    # We will let it run on the full data to ensure robustness.

    # Force re-processing to ensure our config overrides (like paths) are respected
    # and to test the full pipeline.
    if os.path.exists(config.TRAIN_CACHE):
        os.remove(config.TRAIN_CACHE)
    if os.path.exists(config.VAL_CACHE):
        os.remove(config.VAL_CACHE)

    try:
        trainer.run_training(
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE,
            patience=config.PATIENCE,
            load_cached_data=False,  # Force processing
        )
        print("Training loop executed successfully.")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e

    # Check if model checkpoint exists
    if os.path.exists(config.MODEL_CHECKPOINT):
        print(f"Checkpoint found at {config.MODEL_CHECKPOINT}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # =========================================================================
    # 7. Submission Generation Demonstration
    # =========================================================================
    print("\n[7] Generating Submission")

    if os.path.exists(config.TEST_CACHE):
        os.remove(config.TEST_CACHE)

    try:
        trainer.generate_submission(
            batch_size=config.BATCH_SIZE, load_cached_data=False
        )
        print("Submission generation executed successfully.")
    except Exception as e:
        print(f"Submission generation failed: {e}")
        raise e

    if os.path.exists(config.SUBMISSION_FILE):
        print(f"Submission file created at {config.SUBMISSION_FILE}")
        # Verify content
        sub_df = pd.read_csv(config.SUBMISSION_FILE)
        print("Submission head:")
        print(sub_df.head())
        assert len(sub_df) > 0, "Submission file is empty"
        assert all(
            col in sub_df.columns
            for col in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        ), "Submission columns missing"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
