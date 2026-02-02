import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_STATE,
)
from library.data_loader import get_dataset
from library.preprocessing import GNSSScaler, GNSSSequenceDataset
from library.model import GeometryConditionedCNN
from library.utils import meters_to_degrees


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_predictions(load_cached_data: bool = True):
    """
    Main inference function to generate predictions on the test set.

    Args:
        load_cached_data: Whether to load pre-processed test data from cache.
    """
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference using device: {device}")

    # 1. Load Test Data
    # get_dataset handles parquet caching internally based on load_cached_data flag
    test_df = get_dataset("test", load_cached_data=load_cached_data)

    # 2. Load Scaler
    scaler = GNSSScaler()
    scaler_path = os.path.join(WORKING_DIR, "scaler.json")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found at {scaler_path}. Please train the model first."
        )

    print(f"Loading scaler from {scaler_path}")
    scaler.load(scaler_path)

    # 3. Create Dataset and Loader
    # is_test=True ensures we process every timestamp and handle padding if necessary
    test_dataset = GNSSSequenceDataset(test_df, scaler, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Load Model
    model = GeometryConditionedCNN().to(device)
    model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Inference Loop
    all_preds = []
    print("Starting inference loop...")

    with torch.no_grad():
        for i, (traj, ctx) in enumerate(test_loader):
            traj = traj.to(device)
            ctx = ctx.to(device)

            # Forward pass
            # Output shape: [Batch, 2] -> (res_lat_m, res_lon_m)
            output = model(traj, ctx)
            all_preds.append(output.cpu().numpy())

    # Concatenate all predictions
    pred_residuals_m = np.concatenate(all_preds, axis=0)

    # 6. Reconstruction
    # We need to add the predicted residuals (in meters) back to the WLS baseline (in degrees).
    # The test_df order matches the dataset iteration order.

    wls_lat = test_df["wls_lat"].values
    wls_lon = test_df["wls_lon"].values

    pred_res_lat_m = pred_residuals_m[:, 0]
    pred_res_lon_m = pred_residuals_m[:, 1]

    # Convert metric residuals to degrees
    # meters_to_degrees handles the cosine scaling for longitude
    pred_lat_deg, pred_lon_deg = meters_to_degrees(
        pred_res_lat_m, pred_res_lon_m, wls_lat
    )

    final_lat = wls_lat + pred_lat_deg
    final_lon = wls_lon + pred_lon_deg

    # 7. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_df["tripId"],
            "UnixTimeMillis": test_df["UnixTimeMillis"],
            "LatitudeDegrees": final_lat,
            "LongitudeDegrees": final_lon,
        }
    )

    # 8. Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")
