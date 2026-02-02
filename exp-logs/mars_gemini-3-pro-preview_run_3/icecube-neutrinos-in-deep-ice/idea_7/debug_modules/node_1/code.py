import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import (
    load_sensor_geometry,
    direction_to_angles,
    angles_to_direction,
    compute_canonical_frame,
)
from library.dataset import IceCubeDataset
from library.model import DFCGN
from library.loss import CosineSimilarityLoss, get_angular_error
from library.train import train_model
from library.inference import predict_and_submit


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # 1. Initialization
    set_seed(42)
    print("Starting Demo Script...")

    # 2. Configuration Override for Demo Speed
    print("Configuring for fast demo execution...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 200  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2
    Config.WORKING_DIR = Path("./working/demo_run")
    Config.MODEL_SAVE_PATH = Config.WORKING_DIR / "model.pth"
    Config.SUBMISSION_PATH = Config.WORKING_DIR / "submission" / "submission.csv"

    # Ensure clean working state
    if Config.WORKING_DIR.exists():
        import shutil

        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_PATH.parent, exist_ok=True)

    # 3. Utility Verification
    print("\n--- Verifying Utilities ---")

    # 3a. Geometry
    geometry = load_sensor_geometry()
    print(f"Sensor geometry shape: {geometry.shape}")
    assert geometry.shape == (5160, 3), "Geometry shape mismatch"

    # 3b. Angle Conversions
    # Test vector: X-axis [1, 0, 0] -> Azimuth 0, Zenith pi/2
    test_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    az, zen = direction_to_angles(test_vec)

    assert torch.isclose(
        az, torch.tensor([0.0]), atol=1e-4
    ).all(), f"Azimuth mismatch: {az}"
    assert torch.isclose(
        zen, torch.tensor([np.pi / 2]), atol=1e-4
    ).all(), f"Zenith mismatch: {zen}"

    rec_vec = angles_to_direction(az, zen)
    assert torch.allclose(test_vec, rec_vec, atol=1e-4), "Round trip conversion failed"
    print("Angle conversions verified.")

    # 3c. Canonical Frame
    # Create synthetic pulse data
    p_x = np.array([0, 10, 0], dtype=np.float32)
    p_y = np.array([0, 0, 0], dtype=np.float32)
    p_z = np.array([0, 0, 10], dtype=np.float32)
    p_t = np.array([0, 10, 20], dtype=np.float32)
    p_q = np.array([1, 1, 1], dtype=np.float32)

    R = compute_canonical_frame(p_x, p_y, p_z, p_t, p_q)
    det = np.linalg.det(R)
    print(f"Rotation Matrix Determinant: {det:.4f}")
    assert np.isclose(det, 1.0, atol=1e-4), "Rotation matrix must have determinant 1"
    print("Canonical frame computation verified.")

    # 4. Dataset Verification
    print("\n--- Verifying Dataset ---")
    # Initialize dataset (will trigger caching for debug subset)
    train_ds = IceCubeDataset(mode="train")
    print(f"Dataset length (Debug): {len(train_ds)}")
    assert len(train_ds) > 0, "Dataset is empty"

    # Fetch one sample
    sample = train_ds[0]
    print(f"Sample features shape: {sample.x.shape}")
    print(f"Sample target shape: {sample.y.shape}")

    # Check feature dimensions (9 input channels defined in Config)
    assert sample.x.shape[1] == 9, f"Expected 9 features, got {sample.x.shape[1]}"
    # Check target dimensions (1 event, 2 angles)
    assert sample.y.shape == (
        1,
        2,
    ), f"Expected target shape (1, 2), got {sample.y.shape}"
    print("Dataset structure verified.")

    # 5. Model & Loss Verification
    print("\n--- Verifying Model & Loss ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFCGN().to(device)

    # Create a loader to get a batch object (needed for knn_graph inside model)
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    batch = next(iter(loader)).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(batch)

    print(f"Prediction shape: {preds.shape}")
    assert preds.shape == (4, 3), f"Expected output (4, 3), got {preds.shape}"

    # Loss calculation
    criterion = CosineSimilarityLoss()
    loss = criterion(preds, batch.y)
    mae = get_angular_error(preds, batch.y)

    print(f"Loss: {loss.item():.4f}")
    print(f"MAE: {mae:.4f}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert mae >= 0, "MAE should be non-negative"
    print("Model forward pass and loss verified.")

    # 6. Training Pipeline Demo
    print("\n--- Running Training Demo ---")
    # train_model() uses Config globals we modified
    train_model()

    assert Config.MODEL_SAVE_PATH.exists(), "Model checkpoint was not saved."
    print("Training demo completed.")

    # 7. Inference Pipeline Demo
    print("\n--- Running Inference Demo ---")
    # predict_and_submit() uses Config globals
    predict_and_submit()

    assert Config.SUBMISSION_PATH.exists(), "Submission file was not created."

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    expected_cols = ["event_id", "azimuth", "zenith"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), "Missing columns in submission"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference demo completed.")
    print("\nAll demonstrations passed successfully.")
