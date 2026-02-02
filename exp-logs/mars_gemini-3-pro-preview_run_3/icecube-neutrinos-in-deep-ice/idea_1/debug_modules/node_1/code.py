import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import from the provided library
from library.config import Config
from library.utils import (
    load_sensor_geometry,
    angles_to_direction,
    direction_to_angles,
    angular_dist_score,
)
from library.data import IceCubeBatchDataset
from library.model import PointNetBaseline
from library.train import run_training
from library.inference import predict_and_submit


def setup_demo_config():
    """
    Overrides the default Config attributes to ensure the demo runs quickly
    and uses a separate working directory.
    """
    print("Setting up demo configuration...")

    # Use a specific directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load for demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 500  # Only use 500 events
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_PULSES = 32  # Fewer pulses per event
    Config.HIDDEN_DIM = 64  # Smaller model

    # Set device to CPU for simple logic verification if GPU is busy,
    # but use GPU if available for speed.
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Configured for device: {Config.DEVICE}")


def verify_utils():
    """
    Verifies the correctness of utility functions.
    """
    print("\n--- Verifying Utils ---")

    # 1. Test Geometry Loading
    geo = load_sensor_geometry(Config.SENSOR_GEOMETRY_PATH)
    print(f"Sensor geometry shape: {geo.shape}")
    assert geo.shape[1] == 3, "Geometry should have 3 columns (x, y, z)"
    assert len(geo) > 5000, "Should have > 5000 sensors"

    # 2. Test Angle <-> Direction Conversion
    # Test Case: Zenith=0 (Up) -> (0, 0, 1)
    az = torch.tensor([0.0])
    zen = torch.tensor([0.0])
    vec = angles_to_direction(az, zen)

    assert torch.allclose(
        vec, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6
    ), f"Zenith 0 should be (0,0,1), got {vec}"

    # Test Case: Azimuth=0, Zenith=pi/2 (x-axis) -> (1, 0, 0)
    az = torch.tensor([0.0])
    zen = torch.tensor([np.pi / 2])
    vec = angles_to_direction(az, zen)
    assert torch.allclose(
        vec, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6
    ), f"Azimuth 0, Zenith pi/2 should be (1,0,0), got {vec}"

    # Round trip
    az_orig = torch.tensor([1.0, 2.0, 5.0])  # Random angles
    zen_orig = torch.tensor([0.5, 1.5, 2.5])
    vecs = angles_to_direction(az_orig, zen_orig)
    az_rec, zen_rec = direction_to_angles(vecs)

    assert torch.allclose(az_orig, az_rec, atol=1e-5), "Azimuth round-trip failed"
    assert torch.allclose(zen_orig, zen_rec, atol=1e-5), "Zenith round-trip failed"

    # 3. Test Metric
    # Perfect prediction
    y_true = np.array([[1.0, 1.0]])
    y_pred = np.array([[1.0, 1.0]])
    score = angular_dist_score(y_true, y_pred)
    assert score < 1e-6, f"Perfect prediction should have 0 error, got {score}"

    # Orthogonal vectors (error should be pi/2 approx 1.57)
    # (0,0) -> z-axis, (0, pi/2) -> x-axis
    y_true = np.array([[0.0, 0.0]])
    y_pred = np.array([[0.0, np.pi / 2]])
    score = angular_dist_score(y_true, y_pred)
    assert np.isclose(
        score, np.pi / 2, atol=1e-5
    ), f"Orthogonal error should be pi/2, got {score}"

    print("Utils verification passed.")


def verify_dataset_and_model():
    """
    Verifies dataset loading and model forward pass.
    """
    print("\n--- Verifying Dataset and Model ---")

    # Load metadata to find a valid batch ID
    train_meta = pd.read_parquet(Config.TRAIN_META)
    batch_id = train_meta["batch_id"].iloc[0]
    print(f"Testing with Batch ID: {batch_id}")

    sensor_geo = load_sensor_geometry(Config.SENSOR_GEOMETRY_PATH)

    # Instantiate Dataset
    dataset = IceCubeBatchDataset(
        batch_id=batch_id,
        meta_df=train_meta,
        sensor_geo=sensor_geo,
        mode="train",
        load_cached_data=False,  # Force processing
    )

    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print(
            "Dataset is empty (possibly due to filtering in demo). Skipping model check."
        )
        return

    # Check Item Structure
    sample = dataset[0]
    X, y = sample
    print(f"Sample X shape: {X.shape}")  # Should be (NUM_PULSES, 6)
    print(f"Sample y shape: {y.shape}")  # Should be (2,)

    assert X.shape == (
        Config.NUM_PULSES,
        Config.INPUT_DIM,
    ), f"Expected input shape ({Config.NUM_PULSES}, {Config.INPUT_DIM}), got {X.shape}"
    assert y.shape == (2,), f"Expected target shape (2,), got {y.shape}"

    # Instantiate Model
    model = PointNetBaseline(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
    ).to(Config.DEVICE)

    # Run Forward Pass with a mini-batch
    # Create batch of size 4
    X_batch = X.unsqueeze(0).repeat(4, 1, 1).to(Config.DEVICE)  # (4, N, 6)

    model.eval()
    with torch.no_grad():
        preds = model(X_batch)

    print(f"Model output shape: {preds.shape}")
    assert preds.shape == (4, 3), f"Expected output (4, 3), got {preds.shape}"

    # Check normalization logic usually applied after model
    preds_norm = F.normalize(preds, p=2, dim=1)
    norms = torch.norm(preds_norm, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms)), "Normalization failed"

    print("Dataset and Model verification passed.")


def verify_training_pipeline():
    """
    Runs the full training loop for 1 epoch on a subset of data.
    """
    print("\n--- Verifying Training Pipeline ---")

    # Run training
    # This function handles metadata loading, splitting, and the training loop
    run_training(Config)

    # Check if model was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file was not created at {Config.MODEL_PATH}"

    print("Training pipeline verification passed.")


def verify_inference_pipeline():
    """
    Runs the inference loop and checks submission generation.
    """
    print("\n--- Verifying Inference Pipeline ---")

    # Run inference
    predict_and_submit(Config)

    # Check submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    required_cols = ["event_id", "azimuth", "zenith"]
    for col in required_cols:
        assert col in df_sub.columns, f"Missing column {col} in submission"

    assert len(df_sub) > 0, "Submission dataframe is empty"

    # Validate value ranges
    assert df_sub["azimuth"].min() >= 0, "Azimuth contains negative values"
    assert df_sub["azimuth"].max() <= 2 * np.pi, "Azimuth exceeds 2pi"
    assert df_sub["zenith"].min() >= 0, "Zenith contains negative values"
    assert df_sub["zenith"].max() <= np.pi, "Zenith exceeds pi"

    print("Inference pipeline verification passed.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Verify Components
        verify_utils()
        verify_dataset_and_model()

        # 3. Verify Pipelines
        verify_training_pipeline()
        verify_inference_pipeline()

        print("\nAll demonstrations and verifications completed successfully.")

    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        raise e
