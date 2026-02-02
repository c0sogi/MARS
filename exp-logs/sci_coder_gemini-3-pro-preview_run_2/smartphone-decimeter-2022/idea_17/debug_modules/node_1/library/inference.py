import os
import torch
import numpy as np
import pandas as pd
from library.config import SEED, MODEL_PATH, SUBMISSION_DIR, BATCH_SIZE, NUM_WORKERS
from library.model import SCRCNN
from library.dataset import get_dataloaders
from library.utils import get_local_scale_factors


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
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


def generate_submission(
    load_cached_data: bool = True,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    max_batches: int = None,
):
    """
    Generates the submission file using the trained model.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        max_batches (int, optional): Limit the number of batches for debugging.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    print("Loading Test Data...")
    # We unpack the loaders. We only need the test_loader and test_meta.
    # get_dataloaders returns: (train_loader, val_loader, test_loader, test_meta)
    _, _, test_loader, test_meta = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )

    # Initialize Model
    model = SCRCNN().to(device)

    # Load Weights
    if os.path.exists(MODEL_PATH):
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        print(
            f"Warning: No trained model found at {MODEL_PATH}. Predictions will be random (untrained initialization)."
        )

    model.eval()

    preds_list = []

    print("Running inference...")
    with torch.no_grad():
        for i, (x_kin, x_sky) in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                print(f"Debug limit reached: {max_batches} batches.")
                break

            x_kin = x_kin.to(device)
            x_sky = x_sky.to(device)

            # Forward pass
            # Output shape: (Batch, 2) -> [dLat_m, dLon_m]
            outputs = model(x_kin, x_sky)
            preds_list.append(outputs.cpu().numpy())

    if not preds_list:
        print("No predictions generated.")
        return

    # Concatenate predictions
    preds = np.concatenate(preds_list, axis=0)

    # Handle case where we limited batches (slice metadata to match)
    if max_batches is not None:
        n_preds = preds.shape[0]
        test_meta = test_meta.iloc[:n_preds].copy()

    # Reconstruction: Convert metric residuals back to degrees
    # Get baseline WLS positions from metadata
    wls_lats = test_meta["wls_lat"].values
    wls_lons = test_meta["wls_lon"].values

    # Calculate local scale factors (meters per degree)
    # lat_scale: meters per degree latitude
    # lon_scale: meters per degree longitude
    lat_scales, lon_scales = get_local_scale_factors(wls_lats)

    # Convert predicted meters to degrees
    # dDeg = dMeters / Scale
    d_lat_deg = preds[:, 0] / lat_scales
    d_lon_deg = preds[:, 1] / lon_scales

    # Add residuals to baseline WLS
    pred_lats = wls_lats + d_lat_deg
    pred_lons = wls_lons + d_lon_deg

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lats,
            "LongitudeDegrees": pred_lons,
        }
    )

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Save Submission
    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Generated {len(submission)} predictions.")
