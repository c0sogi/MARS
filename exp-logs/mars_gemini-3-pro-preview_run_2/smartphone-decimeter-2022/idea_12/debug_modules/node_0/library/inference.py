import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import SkyContextualizedCNN
from library.data_loader import get_dataloaders
from library.utils import seed_everything, meters_to_latlon


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test set using the trained model.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache via the data loader.
                                 If False, forces the data loader to re-process raw data.
    """
    # Set random seed for reproducibility
    seed_everything(Config.RANDOM_STATE)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # We utilize the shared data loader which handles caching and preprocessing.
    # We only need the test loader and the corresponding metadata.
    print("Loading test data...")
    # get_dataloaders returns: train_loader, val_loader, test_loader, val_meta, test_meta
    _, _, test_loader, _, test_meta = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Load Model
    print(f"Loading model from {Config.MODEL_PATH}...")
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
        )

    model = SkyContextualizedCNN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 3. Inference
    print("Running inference on test set...")
    test_preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch. Test loader returns (traj, sky) as targets are None.
            traj = batch[0].to(device)
            sky = batch[1].to(device)

            # Forward pass
            # Output shape: (Batch, 2) -> [d_east, d_north] in meters
            output = model(traj, sky)

            # Collect predictions
            test_preds_list.append(output.cpu().numpy())

    # Concatenate all predictions into a single array
    test_preds = np.concatenate(test_preds_list, axis=0)

    # 4. Post-processing (Reconstruction)
    print("Reconstructing absolute coordinates...")

    # Get baseline WLS coordinates from the aligned test metadata
    # The data loader ensures test_meta rows correspond 1:1 with the test_loader samples
    t_wls_lat = test_meta["wls_lat"].values
    t_wls_lon = test_meta["wls_lon"].values

    # Predicted residuals are in meters (d_east, d_north)
    pred_d_east = test_preds[:, 0]
    pred_d_north = test_preds[:, 1]

    # Convert metric residuals back to Latitude/Longitude degrees
    # This applies the offset to the WLS baseline
    pred_lat, pred_lon = meters_to_latlon(
        t_wls_lat, t_wls_lon, pred_d_east, pred_d_north
    )

    # 5. Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["utcTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # 6. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
