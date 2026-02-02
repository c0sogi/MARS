import os
import torch
import numpy as np
import pandas as pd

from library.config import (
    MODEL_CHECKPOINT_PATH,
    SUBMISSION_OUTPUT_PATH,
    DEVICE,
    SEED,
)
from library.utils import meters_to_degrees_diff
from library.model import SkyMotionModel
from library.data_loader import get_test_loader


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test set.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # Set random seed
    set_seed()

    # 1. Prepare Test Data
    # This handles caching internally via the data_loader module
    test_loader, test_meta = get_test_loader(load_cached_data=load_cached_data)

    # 2. Load Model
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}"
        )

    print(f"Loading model from {MODEL_CHECKPOINT_PATH}...")
    model = SkyMotionModel()
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Run Inference
    print("Running inference on test set...")
    predictions = []

    with torch.no_grad():
        for traj, sky in test_loader:
            traj = traj.to(DEVICE)
            sky = sky.to(DEVICE)

            # Forward pass
            # Output is (Batch, 2) -> [d_lat_m, d_lon_m]
            outputs = model(traj, sky)
            predictions.append(outputs.cpu().numpy())

    # Concatenate all batches
    pred_residuals = np.vstack(predictions)

    # 4. Post-processing
    print("Reconstructing absolute coordinates...")

    # Extract baseline WLS coordinates from metadata
    wls_lat = test_meta["wls_lat"].values
    wls_lon = test_meta["wls_lon"].values

    # Extract predicted residuals (meters)
    d_lat_m = pred_residuals[:, 0]
    d_lon_m = pred_residuals[:, 1]

    # Convert metric residuals to degree residuals
    # Using the utility function which handles local linear approximation
    d_lat_deg, d_lon_deg = meters_to_degrees_diff(d_lat_m, d_lon_m, wls_lat)

    # Add residuals to baseline to get final prediction
    pred_lat = wls_lat + d_lat_deg
    pred_lon = wls_lon + d_lon_deg

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # 6. Save Submission
    # Ensure parent directory exists (handled by config, but good practice)
    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_OUTPUT_PATH}")
    print(f"Submission shape: {submission_df.shape}")
