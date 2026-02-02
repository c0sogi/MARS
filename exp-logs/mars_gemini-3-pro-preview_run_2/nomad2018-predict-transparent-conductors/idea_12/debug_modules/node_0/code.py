import os
import shutil
import numpy as np
import torch
import pandas as pd
from ase.io import read

# Import library modules
import library.config
import library.utils
import library.data
import library.model
import library.train
import library.predict


def run_demo():
    print("=" * 60)
    print("STARTING LIBRARY DEMO")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Patching
    # -------------------------------------------------------------------------
    print("\n[1] Setting up demo environment...")

    # Define demo directories in ./working
    DEMO_ROOT = "./working/demo_run"
    DEMO_CACHE = os.path.join(DEMO_ROOT, "cache")
    DEMO_CHECKPOINT = os.path.join(DEMO_ROOT, "checkpoints")
    DEMO_SUBMISSION = os.path.join(DEMO_ROOT, "submission")

    # Clean up previous run if exists
    if os.path.exists(DEMO_ROOT):
        shutil.rmtree(DEMO_ROOT)

    os.makedirs(DEMO_CACHE, exist_ok=True)
    os.makedirs(DEMO_CHECKPOINT, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION, exist_ok=True)

    print(f"Created demo directory: {DEMO_ROOT}")

    # Patch module-level variables to redirect outputs and limit data size
    # We patch the variables in the modules where they are used
    library.data.CACHE_DIR = DEMO_CACHE
    library.data.DEBUG_SAMPLE_SIZE = 50  # Limit to 50 samples for speed

    library.train.CHECKPOINT_DIR = DEMO_CHECKPOINT
    library.train.SUBMISSION_DIR = DEMO_SUBMISSION

    library.predict.CACHE_DIR = DEMO_CACHE
    library.predict.CHECKPOINT_DIR = DEMO_CHECKPOINT
    library.predict.SUBMISSION_DIR = DEMO_SUBMISSION

    print("Configuration patched for speed and isolation.")

    # -------------------------------------------------------------------------
    # 2. Utilities Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test LogStandardScaler
    scaler = library.utils.LogStandardScaler()
    dummy_targets = np.random.rand(10, 2).astype(np.float32) * 10  # Random values
    scaler.fit(dummy_targets)
    transformed = scaler.transform(torch.tensor(dummy_targets))
    inverse = scaler.inverse_transform(transformed)

    assert isinstance(
        transformed, torch.Tensor
    ), "Scaler transform should return Tensor"
    assert np.allclose(
        dummy_targets, inverse.numpy(), atol=1e-5
    ), "Scaler inverse transform failed"
    print("  LogStandardScaler: OK")

    # Test Graph Construction
    # Load a real geometry file to test build_pbc_graph
    train_meta = pd.read_csv(library.config.TRAIN_CSV)
    first_file = train_meta.iloc[0]["file_path"]
    full_path = os.path.join(library.config.INPUT_DIR, first_file)

    if os.path.exists(full_path):
        atoms = read(full_path)
        graph_data = library.utils.build_pbc_graph(atoms)

        assert graph_data.x.ndim == 1, "Node features should be 1D (atomic numbers)"
        assert graph_data.edge_index.shape[0] == 2, "Edge index should be (2, E)"
        assert graph_data.edge_attr.shape[1] == 1, "Edge attr should be (E, 1)"
        print(
            f"  Graph Construction: OK (Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges})"
        )
    else:
        print("  Skipping Graph Construction test (file not found)")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Loading Data...")

    # Get dataloaders (this will trigger processing and caching)
    # load_cached_data=False ensures we process the small subset defined by DEBUG_SAMPLE_SIZE
    train_loader, val_loader, test_loader, comp_scaler, target_scaler = (
        library.data.get_dataloaders(load_cached_data=False)
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty"

    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"  Batch structure: {batch}")
    print(f"  Batch composition shape: {batch.composition.shape}")
    print(f"  Batch y shape: {batch.y.shape}")

    # -------------------------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    device = library.config.DEVICE
    model = library.model.ACSRNet().to(device)
    print(f"  Model created on {device}")

    # Move batch to device
    batch = batch.to(device)
    comp_input = comp_scaler.transform(batch.composition)

    # Forward pass
    with torch.no_grad():
        output = model(comp_input, batch)

    print(f"  Forward pass output shape: {output.shape}")
    assert output.shape == (
        batch.num_graphs,
        2,
    ), f"Expected output shape {(batch.num_graphs, 2)}, got {output.shape}"
    print("  Forward pass: OK")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demo
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs)...")

    # We use the run_training function from library.train
    # It handles optimization, loss, and saving checkpoints
    library.train.run_training(
        num_epochs=2, patience=1, load_cached_data=True  # Use the cache we just created
    )

    expected_checkpoint = os.path.join(DEMO_CHECKPOINT, "best_model.pth")
    if os.path.exists(expected_checkpoint):
        print(f"  Checkpoint created successfully: {expected_checkpoint}")
    else:
        raise FileNotFoundError("Checkpoint file was not created during training.")

    # -------------------------------------------------------------------------
    # 6. Prediction Demo
    # -------------------------------------------------------------------------
    print("\n[6] Generating Predictions...")

    # We use generate_predictions from library.predict
    # It loads the best model and generates the submission file
    # We set debug_sample_size again to ensure consistency if it re-processes
    library.predict.generate_predictions(
        load_cached_data=True, debug_sample_size=library.data.DEBUG_SAMPLE_SIZE
    )

    expected_submission = os.path.join(DEMO_SUBMISSION, "submission.csv")
    if os.path.exists(expected_submission):
        print(f"  Submission file created successfully: {expected_submission}")

        # Verify submission format
        df_sub = pd.read_csv(expected_submission)
        print("  Submission head:")
        print(df_sub.head())

        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
