import os
import sys
import numpy as np
import torch
import pandas as pd
from unittest.mock import patch

# Import library modules
from library.config import (
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    ATOMIC_INPUT_DIM,
    GLOBAL_INPUT_DIM,
    DEVICE,
)
from library.utils import (
    set_seed,
    compute_cell_volume,
    cartesian_to_fractional,
    get_pbc_distances,
)
from library.data import get_dataloaders
from library.model import EWADeepSets
from library.train import Trainer


def test_utils():
    """
    Validates utility functions for geometry calculations.
    """
    print("\n--- Testing Utils ---")

    # 1. Test compute_cell_volume
    # A 10x10x10 cubic cell should have volume 1000
    vol = compute_cell_volume(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    print(f"Computed volume (10x10x10 cube): {vol}")
    assert np.isclose(vol, 1000.0), f"Volume calculation failed: {vol}"

    # 2. Test cartesian_to_fractional
    # Point [5,5,5] in a 10x10x10 box should be [0.5, 0.5, 0.5]
    lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    coords = np.array([[5.0, 5.0, 5.0]])
    frac = cartesian_to_fractional(coords, lattice)
    print(f"Fractional coords of [5,5,5] in 10x10x10: {frac}")
    assert np.allclose(frac, 0.5), f"Fractional conversion failed: {frac}"

    # 3. Test get_pbc_distances
    # Two atoms at x=1 and x=9 in a box of length 10.
    # The Euclidean distance is 8, but PBC distance should be 2.
    coords_pbc = np.array([[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]])
    dists = get_pbc_distances(coords_pbc, lattice)
    print(f"PBC Nearest Neighbor Distances: {dists.flatten()}")

    # Atom 0 is 2 units away from Atom 1 (via boundary)
    # Atom 1 is 2 units away from Atom 0 (via boundary)
    assert np.allclose(dists, 2.0), f"PBC distance failed: {dists}"
    print("Utils tests passed.")


def test_data_and_model_pipeline():
    """
    Demonstrates the full pipeline: Data Loading -> Model Init -> Training -> Prediction.
    Uses a mocked subset of data for speed.
    """
    print("\n--- Testing Data Pipeline and Model ---")

    # Mock pandas.read_csv to read only a small subset of data (first 10 rows)
    # This ensures the demonstration runs quickly without processing the entire dataset.
    original_read_csv = pd.read_csv

    def fast_read_csv(filepath, *args, **kwargs):
        print(f"Mock reading {filepath} (subsetting to 10 rows)")
        df = original_read_csv(filepath, *args, **kwargs)
        return df.head(10)

    # Apply the mock
    with patch("pandas.read_csv", side_effect=fast_read_csv):

        # 1. Get DataLoaders
        # load_cached_data=False forces the data processing logic to run on our subset
        print("Initializing DataLoaders with subset...")
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

        # Inspect a single batch
        batch = next(iter(train_loader))
        atomic_feats = batch["atomic_features"]
        global_feats = batch["global_features"]
        targets = batch["targets"]
        mask = batch["mask"]

        print(f"\nBatch keys: {batch.keys()}")
        print(
            f"Atomic features shape: {atomic_feats.shape} (Expected: [B, N, {ATOMIC_INPUT_DIM}])"
        )
        print(
            f"Global features shape: {global_feats.shape} (Expected: [B, {GLOBAL_INPUT_DIM}])"
        )
        print(f"Targets shape: {targets.shape} (Expected: [B, 2])")

        # Verify dimensions match configuration
        assert atomic_feats.shape[2] == ATOMIC_INPUT_DIM
        assert global_feats.shape[1] == GLOBAL_INPUT_DIM
        assert targets.shape[1] == 2

        # 2. Instantiate Model
        print("\nInstantiating EWADeepSets Model...")
        model = EWADeepSets()
        # Ensure model is on the correct device (CPU for demonstration if CUDA not forced)
        model.to(DEVICE)

        # 3. Forward Pass
        print("Running Forward Pass...")
        # Move batch inputs to the same device as model
        outputs = model(
            atomic_feats.to(DEVICE), mask.to(DEVICE), global_feats.to(DEVICE)
        )
        print(f"Output shape: {outputs.shape}")

        assert outputs.shape == targets.shape
        print("Forward pass successful.")

        # 4. Training Loop (1 Epoch)
        print("\nTesting Training Loop (1 Epoch)...")
        trainer = Trainer(model)

        # Train for 1 epoch
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        assert np.isfinite(train_loss), "Training loss is not finite"

        # Validate
        val_loss, val_rmsle = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.4f}, Val RMSLE: {val_rmsle:.4f}")

        # 5. Prediction
        print("\nTesting Prediction on Test Set...")
        ids, preds = trainer.predict(test_loader)

        print(f"Predictions shape: {preds.shape}")
        print(f"Sample predictions (first 3):\n{preds[:3]}")

        # Since we subsetted to 10 rows, we expect 10 predictions
        assert len(ids) == 10, f"Expected 10 test predictions, got {len(ids)}"
        assert preds.shape == (10, 2)
        print("Prediction successful.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    try:
        test_utils()
        test_data_and_model_pipeline()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise e
