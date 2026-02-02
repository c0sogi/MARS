import os
import sys
import numpy as np
import torch
import pandas as pd
import shutil

# Import the provided library modules
from library import config, utils, features, data, model, engine


def main():
    print("Initializing demonstration...")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    # Override config parameters for a fast demonstration run
    config.MAX_SAMPLES = 50  # Process only 50 samples
    config.NUM_EPOCHS = 2  # Train for only 2 epochs
    config.BATCH_SIZE = 16  # Small batch size
    config.PATIENCE = 1  # Short patience

    # Ensure working directories are clean or exist
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds via engine
    engine.set_seed(config.SEED)
    print("Configuration updated for fast demonstration.")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\nVerifying utility functions...")

    # Test get_unit_cell_volume
    # Simple cubic cell 10x10x10, 90 degrees
    lengths = [10.0, 10.0, 10.0]
    angles = [90.0, 90.0, 90.0]
    vol = utils.get_unit_cell_volume(lengths, angles)
    assert np.isclose(
        vol, 1000.0
    ), f"Volume calculation failed: expected 1000.0, got {vol}"
    print("  - get_unit_cell_volume: Passed")

    # Test calculate_pbc_distances
    # 10x10x10 box. Point A at (1,1,1), Point B at (9,1,1).
    # Distance should be 2.0 due to PBC (1 -> 9 wraps around 10).
    lattice = np.diag([10.0, 10.0, 10.0])
    coords = np.array([[1.0, 1.0, 1.0], [9.0, 1.0, 1.0]])
    dists = utils.calculate_pbc_distances(coords, lattice)
    # dists is (2, 2) matrix
    d_ab = dists[0, 1]
    assert np.isclose(
        d_ab, 2.0
    ), f"PBC distance calculation failed: expected 2.0, got {d_ab}"
    print("  - calculate_pbc_distances: Passed")

    # ==========================================
    # 3. Data Processing
    # ==========================================
    print("\nProcessing data (this involves feature extraction)...")

    # Initialize processor
    processor = data.DataProcessor()

    # Process data and get loaders
    # We set load_cached_data=False to force computation for this demo
    train_loader, val_loader, test_loader = processor.process_and_get_loaders(
        load_cached_data=False
    )

    print(f"  - Train loader size: {len(train_loader)} batches")
    print(f"  - Val loader size: {len(val_loader)} batches")
    print(f"  - Test loader size: {len(test_loader)} batches")

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    atoms, batch_vec, glob_feats, targets, ids = sample_batch

    # Check dimensions
    # atoms: (Total_Atoms, ATOMIC_INPUT_DIM)
    assert (
        atoms.shape[1] == config.ATOMIC_INPUT_DIM
    ), f"Atomic feature dim mismatch. Expected {config.ATOMIC_INPUT_DIM}, got {atoms.shape[1]}"

    # glob_feats: (Batch_Size, GLOBAL_INPUT_DIM)
    assert (
        glob_feats.shape[1] == config.GLOBAL_INPUT_DIM
    ), f"Global feature dim mismatch. Expected {config.GLOBAL_INPUT_DIM}, got {glob_feats.shape[1]}"

    # targets: (Batch_Size, 2)
    assert (
        targets.shape[1] == 2
    ), f"Target dim mismatch. Expected 2, got {targets.shape[1]}"

    print("  - Data batch shapes verified.")

    # ==========================================
    # 4. Model Instantiation & Forward Pass
    # ==========================================
    print("\nInstantiating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.IDCR_WDS_Model().to(device)

    print("  - Performing dummy forward pass...")
    atoms = atoms.to(device)
    batch_vec = batch_vec.to(device)
    glob_feats = glob_feats.to(device)

    with torch.no_grad():
        outputs = net(atoms, batch_vec, glob_feats)

    assert outputs.shape == (
        atoms.shape[0] if batch_vec is None else glob_feats.shape[0],
        2,
    ), f"Output shape mismatch. Expected ({glob_feats.shape[0]}, 2), got {outputs.shape}"
    print("  - Forward pass successful.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\nStarting training loop...")
    trainer = engine.Trainer(net)

    # Run training
    trainer.train(
        train_loader, val_loader, num_epochs=config.NUM_EPOCHS, patience=config.PATIENCE
    )

    # Check if model checkpoint was saved
    if not os.path.exists(config.MODEL_PATH):
        # In a very short run with random init, validation loss might not improve.
        # For the sake of the demo, we force save if not present, or just warn.
        # However, Trainer usually saves if val_loss < inf (which is always true for first epoch).
        # Let's check.
        print(
            "  - Warning: Best model checkpoint not found (might be due to very short training)."
        )
        # Save manually for the next step to work
        torch.save(net.state_dict(), config.MODEL_PATH)
        print("  - Manually saved model for inference step.")
    else:
        print("  - Best model checkpoint found.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\nGenerating submission...")
    engine.generate_submission(net, test_loader, device, config.SUBMISSION_PATH)

    if os.path.exists(config.SUBMISSION_PATH):
        print(f"  - Submission file created at {config.SUBMISSION_PATH}")

        # Verify submission format
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Submission columns mismatch. Got {sub_df.columns}"
        assert len(sub_df) > 0, "Submission file is empty."
        print("  - Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
