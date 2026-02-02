import os
import torch
import numpy as np
import pandas as pd
from library.config import config
from library.model import ECM_MLP
from library.dataset import get_test_dataloader
from library.utils import meters_to_degrees


def generate_submission():
    """
    Generates the submission file by running inference on the test set.

    Steps:
    1. Loads the test DataLoader and raw test DataFrame (for WLS baselines).
    2. Loads the trained ECM_MLP model weights.
    3. Predicts metric residuals (North, East) for each timestamp.
    4. Converts metric residuals to degree offsets.
    5. Adds offsets to WLS baselines to get final predictions.
    6. Saves the result to submission.csv.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # 2. Load Test Data
    # get_test_dataloader returns the loader and the raw dataframe (X_test)
    # X_test contains the 'wls_lat' and 'wls_lon' columns needed for reconstruction
    test_loader, X_test = get_test_dataloader(
        batch_size=config.BATCH_SIZE, num_workers=4
    )

    # 3. Load Model
    model = ECM_MLP().to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Run Inference
    print("Running inference...")
    predictions = []

    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)

            # Forward pass
            # Output shape: (batch_size, 2) -> [delta_north_m, delta_east_m]
            outputs = model(features)

            predictions.append(outputs.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)

    pred_north_m = predictions[:, 0]
    pred_east_m = predictions[:, 1]

    print(f"Predictions shape: {predictions.shape}")

    # 5. Reconstruct Absolute Coordinates
    # We need the baseline WLS positions from the raw test dataframe
    # Ensure alignment: test_loader is not shuffled, so it aligns with X_test rows
    if len(X_test) != len(predictions):
        raise ValueError(
            f"Mismatch between test data size ({len(X_test)}) and predictions ({len(predictions)})"
        )

    print("Reconstructing coordinates...")

    # Extract baseline WLS coordinates
    # These columns are created in library.preprocessing.compute_dynamics
    wls_lat = X_test["wls_lat"].values
    wls_lon = X_test["wls_lon"].values

    # Convert metric residuals to degrees
    # Note: We use the WLS latitude as the reference latitude for longitude scaling
    delta_lat_deg, delta_lon_deg = meters_to_degrees(pred_north_m, pred_east_m, wls_lat)

    # Apply corrections
    final_lat = wls_lat + delta_lat_deg
    final_lon = wls_lon + delta_lon_deg

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": X_test["tripId"],
            "UnixTimeMillis": X_test["utcTimeMillis"],
            "LatitudeDegrees": final_lat,
            "LongitudeDegrees": final_lon,
        }
    )

    # 7. Save Submission
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
