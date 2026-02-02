import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from library.config import (
    MODEL_SAVE_PATH,
    SUBMISSION_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    WORK_DIR,
)
from library.data import load_dataset
from library.model import ResidualMLP
from library.utils import ecef_to_geodetic


def predict_and_submit(load_cached_data=True):
    """
    Loads the trained model and test data, generates predictions, and creates the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed test features from cache.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # 2. Load Test Data
    print("Loading test dataset...")
    test_dataset = load_dataset(mode="test", load_cached_data=load_cached_data)

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # 3. Load Model
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_SAVE_PATH}. Please run training first."
        )

    # Determine input dimension from dataset
    sample_features = test_dataset[0]
    input_dim = sample_features.shape[0]

    model = ResidualMLP(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 4. Inference Loop
    print("Starting inference...")
    all_residuals = []

    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_residuals.append(outputs.cpu().numpy())

    # Concatenate all batch predictions
    # Shape: (N, 3) -> [dx, dy, dz]
    residuals = np.concatenate(all_residuals, axis=0)

    # 5. Reconstruction
    print("Reconstructing positions...")
    # Get metadata which contains the baseline WLS positions
    meta_df = test_dataset.meta.copy()

    # Extract baseline ECEF coordinates
    wls_x = meta_df["WlsPositionXEcefMeters"].values
    wls_y = meta_df["WlsPositionYEcefMeters"].values
    wls_z = meta_df["WlsPositionZEcefMeters"].values

    # Add predicted residuals
    pred_x = wls_x + residuals[:, 0]
    pred_y = wls_y + residuals[:, 1]
    pred_z = wls_z + residuals[:, 2]

    # 6. Coordinate Conversion (ECEF to Geodetic)
    print("Converting ECEF to Geodetic coordinates...")
    pred_lat, pred_lon, _ = ecef_to_geodetic(pred_x, pred_y, pred_z)

    # 7. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": meta_df["tripId"],
            "UnixTimeMillis": meta_df["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Save submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())
