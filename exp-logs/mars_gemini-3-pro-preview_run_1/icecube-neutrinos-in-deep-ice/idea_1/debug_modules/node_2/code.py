import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config, set_seed
from library.utils import (
    load_sensor_geometry,
    angles_to_vector,
    vector_to_angles,
    angular_dist_score,
)
from library.data import NeutrinoDataset, get_dataloaders
from library.model import TemporalCNN, cosine_similarity_loss
from library.train import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting Demonstration of Neutrino Direction Prediction Pipeline ===\n")

    # 1. Setup and Reproducibility
    print("--- Step 1: Configuration and Seeding ---")
    set_seed(42)

    # Configure for speed (Debug mode)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 200  # Small subset for quick validation
    Config.BATCH_SIZE = 16
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(
        f"Configuration set: DEBUG={Config.DEBUG}, SUBSET={Config.DEBUG_SUBSET_SIZE}, EPOCHS={Config.NUM_EPOCHS}"
    )
    print("Seeding complete.\n")

    # 2. Verify Utility Functions
    print("--- Step 2: Verifying Utility Functions ---")

    # Test Geometry Loading
    geo_df = load_sensor_geometry()
    print(f"Geometry loaded. Shape: {geo_df.shape}")
    assert geo_df.shape[1] == 3, "Geometry should have 3 columns (x, y, z)"
    assert len(geo_df) >= 5160, "Geometry should contain at least 5160 sensors"

    # Test Coordinate Transforms
    # Test case: Azimuth=0, Zenith=pi/2 (90 deg) -> Vector along X-axis (1, 0, 0)
    az_test = np.array([0.0])
    zen_test = np.array([np.pi / 2])
    vec = angles_to_vector(az_test, zen_test)

    print(f"Angle to Vector Test: Az=0, Zen=pi/2 -> Vec={vec[0]}")
    assert np.allclose(vec, [[1.0, 0.0, 0.0]], atol=1e-6), "Vector conversion failed"

    # Test Inverse Transform
    az_rec, zen_rec = vector_to_angles(vec)
    print(f"Vector to Angle Test: Vec={vec[0]} -> Az={az_rec[0]}, Zen={zen_rec[0]}")
    assert np.allclose(az_rec, az_test, atol=1e-6), "Azimuth reconstruction failed"
    assert np.allclose(zen_rec, zen_test, atol=1e-6), "Zenith reconstruction failed"

    # Test Angular Distance Score
    # Distance between (1,0,0) and (0,1,0) (90 degrees or pi/2 radians)
    az_p = np.array([np.pi / 2])
    zen_p = np.array([np.pi / 2])
    score = angular_dist_score(az_test, zen_test, az_p, zen_p)
    print(f"Angular Distance Test: 90 deg separation -> Score={score:.4f} rad")
    assert np.isclose(
        score, np.pi / 2, atol=1e-4
    ), "Angular distance calculation failed"
    print("Utility functions verified.\n")

    # 3. Verify Data Loading
    print("--- Step 3: Verifying Data Loading ---")

    # Load metadata manually to inspect
    train_meta = pd.read_parquet(Config.TRAIN_META_PATH).iloc[:10]
    print(f"Loaded {len(train_meta)} meta rows for inspection.")

    # Initialize Dataset
    dataset = NeutrinoDataset(train_meta, geo_df, mode="train")

    # Fetch single item
    x, y = dataset[0]
    print(f"Single Item - Input Shape: {x.shape}, Target Shape: {y.shape}")

    # Validation
    assert x.shape == (
        Config.SEQ_LEN,
        Config.NUM_FEATURES,
    ), f"Input shape mismatch. Expected ({Config.SEQ_LEN}, {Config.NUM_FEATURES})"
    assert y.shape == (3,), "Target shape mismatch. Expected (3,) for 3D vector"
    assert isinstance(x, torch.Tensor), "Input should be a torch Tensor"
    assert isinstance(y, torch.Tensor), "Target should be a torch Tensor"

    # Check DataLoaders
    train_loader, val_loader = get_dataloaders()
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch - Input Shape: {batch_x.shape}, Target Shape: {batch_y.shape}")
    assert batch_x.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    print("Data loading verified.\n")

    # 4. Verify Model Architecture
    print("--- Step 4: Verifying Model Architecture ---")

    model = TemporalCNN()
    model.eval()

    # Forward pass with dummy batch
    with torch.no_grad():
        output = model(batch_x)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 3), "Model output shape mismatch"

    # Loss calculation
    loss = cosine_similarity_loss(output, batch_y)
    print(f"Initial Loss (Cosine Similarity): {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    print("Model architecture verified.\n")

    # 5. Run Training Pipeline
    print("--- Step 5: Running Training Pipeline (Trainer) ---")

    trainer = Trainer()
    # Run a very short training loop
    trainer.run(num_epochs=1, debug=True, subset_size=Config.DEBUG_SUBSET_SIZE)

    # Check if best model was saved
    best_model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), f"Best model not found at {best_model_path}"
    print(f"Training complete. Model saved to {best_model_path}.\n")

    # 6. Verify Submission Generation
    print("--- Step 6: Verifying Submission Generation ---")

    # Check if submission file exists (generated by trainer.run)
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Inspect submission
    sub_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")
    print(f"Columns: {list(sub_df.columns)}")

    assert "event_id" in sub_df.columns, "Missing event_id column"
    assert "azimuth" in sub_df.columns, "Missing azimuth column"
    assert "zenith" in sub_df.columns, "Missing zenith column"
    assert len(sub_df) > 0, "Submission file is empty"

    # Verify values are within valid ranges
    assert sub_df["azimuth"].min() >= 0, "Azimuth contains negative values"
    assert sub_df["azimuth"].max() <= 2 * np.pi, "Azimuth exceeds 2*pi"
    assert sub_df["zenith"].min() >= 0, "Zenith contains negative values"
    assert sub_df["zenith"].max() <= np.pi, "Zenith exceeds pi"

    print("Submission verified.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
