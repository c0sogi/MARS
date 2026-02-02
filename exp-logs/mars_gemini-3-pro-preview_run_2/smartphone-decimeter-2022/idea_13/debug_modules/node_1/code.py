import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Force reload of library modules to pick up changes in persistent environment
# (Cite debug_lesson_3)
for module in [
    "library.utils",
    "library.preprocessing",
    "library.dataset",
    "library.model",
    "library.engine",
    "library.config",
]:
    if module in sys.modules:
        del sys.modules[module]

# Import from the provided library
from library import config
from library.utils import lla_to_ecef, wgs84_to_enu
from library.dataset import get_dataloader
from library.model import MultiScaleKinematicCNN
from library.engine import fit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_utils():
    """Verifies utility functions."""
    print("Testing utility functions...")

    # Test LLA to ECEF (Approximate check)
    # San Francisco coordinates
    lat, lon, alt = 37.7749, -122.4194, 0
    x, y, z = lla_to_ecef(lat, lon, alt)

    # Expected approximate ECEF for SF
    # X ~ -2700km, Y ~ -4290km, Z ~ 3850km
    assert -2.8e6 < x < -2.6e6, f"ECEF X out of range: {x}"
    assert -4.4e6 < y < -4.2e6, f"ECEF Y out of range: {y}"
    assert 3.8e6 < z < 4.0e6, f"ECEF Z out of range: {z}"

    # Test WGS84 to ENU (Relative)
    # Point A
    lat0, lon0, alt0 = 37.7749, -122.4194, 0
    # Point B (slightly north)
    lat1, lon1, alt1 = 37.7750, -122.4194, 0

    e, n, u = wgs84_to_enu(lat1, lon1, alt1, lat0, lon0, alt0)

    # Should be approx 11.1 meters North (1 degree lat ~ 111km -> 0.0001 deg ~ 11.1m)
    assert abs(e) < 1.0, f"Expected East ~ 0, got {e}"
    assert 10.0 < n < 12.0, f"Expected North ~ 11.1, got {n}"
    assert abs(u) < 1.0, f"Expected Up ~ 0, got {u}"

    print("Utils verified.")


def run_demonstration():
    """Runs the main demonstration pipeline."""
    set_seed(config.RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. Data Loading
    # ---------------------------------------------------------
    print("\nInitializing DataLoaders...")

    # We use a very small subset (max_samples=100) and force processing from scratch
    # to demonstrate the pipeline without relying on pre-existing cache for this specific run.
    train_loader = get_dataloader(
        split="train",
        batch_size=16,
        shuffle=True,
        load_cached_data=False,
        max_samples=100,
        num_workers=0,  # Avoid multiprocessing overhead for small sample
    )

    val_loader = get_dataloader(
        split="validation",
        batch_size=16,
        shuffle=False,
        load_cached_data=False,
        max_samples=50,
        num_workers=0,
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    kin_seq = sample_batch["kinematic_sequence"]
    ctx_feats = sample_batch["context_features"]
    targets = sample_batch["target_residual"]

    print(f"Sample batch keys: {sample_batch.keys()}")
    print(
        f"Kinematic sequence shape: {kin_seq.shape}"
    )  # Expected: (B, Window, Features)
    print(f"Context features shape: {ctx_feats.shape}")  # Expected: (B, Features)
    print(f"Targets shape: {targets.shape}")  # Expected: (B, 2)

    # Assertions for shapes
    assert (
        len(kin_seq.shape) == 3
    ), "Kinematic sequence should be 3D (Batch, Time, Feat)"
    assert (
        kin_seq.shape[1] == config.WINDOW_SIZE
    ), f"Window size mismatch. Expected {config.WINDOW_SIZE}, got {kin_seq.shape[1]}"
    assert kin_seq.shape[2] == len(
        config.KINEMATIC_FEATURES
    ), "Kinematic feature count mismatch"
    assert ctx_feats.shape[1] == len(
        config.CONTEXT_FEATURES
    ), "Context feature count mismatch"
    assert targets.shape[1] == 2, "Target should have 2 dimensions (East, North)"

    # ---------------------------------------------------------
    # 2. Model Instantiation
    # ---------------------------------------------------------
    print("\nInstantiating Model...")
    model = MultiScaleKinematicCNN().to(device)

    # Verify forward pass
    print("Verifying forward pass...")
    with torch.no_grad():
        output = model(kin_seq.to(device), ctx_feats.to(device))

    print(f"Output shape: {output.shape}")
    assert output.shape == (kin_seq.shape[0], 2), "Output shape mismatch"
    print("Forward pass successful.")

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    print("\nStarting Training Demonstration (1 Epoch)...")

    # We use library.engine.fit which implements the training loop
    # We reduce patience to 1 and epochs to 1 for speed
    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,
        patience=1,
    )

    print("Training demonstration complete.")

    # ---------------------------------------------------------
    # 4. Inference / Prediction
    # ---------------------------------------------------------
    print("\nRunning Inference on Validation Set...")
    trained_model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for batch in val_loader:
            k = batch["kinematic_sequence"].to(device)
            c = batch["context_features"].to(device)
            t = batch["target_residual"].numpy()

            out = trained_model(k, c).cpu().numpy()
            predictions.append(out)
            ground_truth.append(t)

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    mae = np.mean(np.abs(predictions - ground_truth))
    print(f"Validation MAE (Meters): {mae:.4f}")

    print("\nDemonstration finished successfully.")


if __name__ == "__main__":
    try:
        test_utils()
        run_demonstration()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
