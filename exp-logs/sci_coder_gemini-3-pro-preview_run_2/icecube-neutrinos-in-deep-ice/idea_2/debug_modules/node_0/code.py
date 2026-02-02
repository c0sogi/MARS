import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library
from library.config import (
    TRAIN_META_PATH,
    TEST_META_PATH,
    INPUT_DIM,
    N_PULSES,
    DEVICE,
)
from library.utils import (
    seed_everything,
    load_sensor_geometry,
    vector_to_angles,
)
from library.dataset import IceCubeDataset
from library.model import (
    GeometricPulseAggregator,
    CosineDistanceLoss,
    calculate_angular_error,
)
from library.train import run_training
from library.inference import predict_test_set


def main():
    print("=== Starting Demonstration Script ===\n")

    # 1. Setup
    seed_everything(42)
    working_dir = "./working/demo_run"
    os.makedirs(working_dir, exist_ok=True)

    # 2. Verify Utilities
    print("--- Verifying Utilities ---")

    # 2.1 Sensor Geometry
    geo_df = load_sensor_geometry()
    print(f"Sensor geometry loaded. Shape: {geo_df.shape}")
    assert geo_df.shape[0] == 5160, "Expected 5160 sensors in geometry"
    assert (
        "x" in geo_df.columns and "z" in geo_df.columns
    ), "Geometry missing coordinates"

    # 2.2 Vector to Angles conversion
    # Test case: Vector along X-axis (1, 0, 0) -> Azimuth 0, Zenith pi/2
    test_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    az, ze = vector_to_angles(test_vec)

    print(f"Test Vector (1,0,0) -> Azimuth: {az.item():.4f}, Zenith: {ze.item():.4f}")
    assert np.isclose(
        az.item(), 0.0, atol=1e-5
    ), "Azimuth calculation incorrect for X-axis"
    assert np.isclose(
        ze.item(), np.pi / 2, atol=1e-5
    ), "Zenith calculation incorrect for X-axis"

    # Test case: Vector along Z-axis (0, 0, 1) -> Zenith 0
    test_vec_z = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    _, ze_z = vector_to_angles(test_vec_z)
    assert np.isclose(
        ze_z.item(), 0.0, atol=1e-5
    ), "Zenith calculation incorrect for Z-axis"
    print("Vector to Angles conversion verified.\n")

    # 3. Verify Dataset
    print("--- Verifying Dataset ---")
    subset_size = 50
    dataset = IceCubeDataset(
        metadata_path=TRAIN_META_PATH, mode="train", debug_subset_size=subset_size
    )

    print(f"Dataset initialized with subset size: {len(dataset)}")
    assert len(dataset) == subset_size, "Dataset subset size mismatch"

    # Fetch one sample
    features, target = dataset[0]
    print(f"Sample 0 Features Shape: {features.shape}")
    print(f"Sample 0 Target Shape: {target.shape}")

    # Check shapes
    # Features: (N_PULSES, 6) -> [x, y, z, time, charge, auxiliary]
    assert features.shape == (
        N_PULSES,
        INPUT_DIM,
    ), f"Expected feature shape ({N_PULSES}, {INPUT_DIM})"
    assert target.shape == (3,), "Expected target shape (3,)"

    # Check data types
    assert features.dtype == torch.float32, "Features should be float32"
    assert target.dtype == torch.float32, "Target should be float32"
    print("Dataset verification successful.\n")

    # 4. Verify Model
    print("--- Verifying Model Architecture ---")
    model = GeometricPulseAggregator()
    model.eval()

    # Create a dummy batch: (Batch=2, N_PULSES, INPUT_DIM)
    dummy_input = torch.randn(2, N_PULSES, INPUT_DIM)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 3), "Model output shape mismatch, expected (Batch, 3)"
    print("Model architecture verified.\n")

    # 5. Verify Loss
    print("--- Verifying Loss Function ---")
    criterion = CosineDistanceLoss()

    # Perfect prediction
    pred_perfect = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    target_perfect = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    loss_perfect = criterion(pred_perfect, target_perfect)

    print(f"Loss for perfect prediction: {loss_perfect.item():.6f}")
    assert np.isclose(
        loss_perfect.item(), 0.0, atol=1e-6
    ), "Loss should be 0 for perfect match"

    # Orthogonal prediction (loss should be 1.0)
    pred_ortho = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    loss_ortho = criterion(pred_ortho, target_perfect)
    print(f"Loss for orthogonal prediction: {loss_ortho.item():.6f}")
    assert np.isclose(
        loss_ortho.item(), 1.0, atol=1e-6
    ), "Loss should be 1 for orthogonal vectors"

    # Angular Error Check
    ang_err = calculate_angular_error(pred_ortho, target_perfect)
    print(f"Angular Error (90 deg): {ang_err:.4f} rad")
    assert np.isclose(
        ang_err, np.pi / 2, atol=1e-4
    ), "Angular error calculation incorrect"
    print("Loss function verified.\n")

    # 6. Verify Training Pipeline
    print("--- Running Training Pipeline Demonstration ---")
    # We use a very small subset and 1 epoch to keep it fast
    train_subset_size = 100
    batch_size = 16
    epochs = 1

    print(f"Training on {train_subset_size} events for {epochs} epoch...")

    run_training(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        patience=1,
        debug_subset_size=train_subset_size,
        save_dir=working_dir,
    )

    model_path = os.path.join(working_dir, "best_model.pth")
    assert os.path.exists(model_path), "Training failed to save best_model.pth"
    print(f"Training complete. Model saved to {model_path}\n")

    # 7. Verify Inference Pipeline
    print("--- Running Inference Pipeline Demonstration ---")
    test_subset_size = 50
    submission_path = os.path.join(working_dir, "submission.csv")

    print(f"Predicting on {test_subset_size} test events...")

    predict_test_set(
        model_path=model_path,
        output_path=submission_path,
        batch_size=batch_size,
        num_workers=2,
        device=DEVICE,
        debug_subset_size=test_subset_size,
    )

    assert os.path.exists(
        submission_path
    ), "Inference failed to generate submission.csv"

    # Validate submission format
    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {sub_df.shape}")

    assert sub_df.shape == (test_subset_size, 3), "Submission shape mismatch"
    assert list(sub_df.columns) == [
        "event_id",
        "azimuth",
        "zenith",
    ], "Submission columns mismatch"
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"

    print("Inference pipeline verified.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
