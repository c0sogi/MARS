import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.model import SensorFusionTCN
from library.data_loader import SmartphoneDataset


def generate_submission(
    test_meta_path=Config.TEST_METADATA_PATH,
    model_weights_path=os.path.join(Config.WORKING_DIR, "model_weights.pth"),
    output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Generates the submission file using the trained model and test metadata.

    This function performs the following steps:
    1. Loads the test dataset using SmartphoneDataset, which handles data alignment,
       windowing, and caching.
    2. Initializes the SensorFusionTCN model and loads trained weights.
    3. Runs inference to predict coordinate residuals (Delta Lat, Delta Lon).
    4. Reconstructs the final Latitude and Longitude by adding residuals to the
       Weighted Least Squares (WLS) baseline provided in the dataset.
    5. Formats the output to match the competition submission requirement and saves it.

    Args:
        test_meta_path (str): Path to the test metadata CSV file.
        model_weights_path (str): Path to the saved model weights (.pth file).
        output_path (str): Destination path for the generated submission CSV.
        batch_size (int): Batch size for the data loader during inference.
        device (str): Computation device ('cpu' or 'cuda').
    """
    print(f"Generating submission using metadata from {test_meta_path}")

    # 1. Load Test Dataset
    # SmartphoneDataset with mode='test' filters for rows requiring prediction (valid tripId)
    # and returns (features, wls_baseline) tuples.
    test_dataset = SmartphoneDataset(
        metadata_path=test_meta_path, window_size=Config.WINDOW_SIZE, mode="test"
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 2. Initialize Model
    model = SensorFusionTCN(
        num_inputs=Config.NUM_FEATURES,
        num_channels=[Config.HIDDEN_CHANNELS] * Config.NUM_LAYERS,
        kernel_size=Config.KERNEL_SIZE,
        dropout=Config.DROPOUT,
    ).to(device)

    # Load weights
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Loaded model weights from {model_weights_path}")
    else:
        print(
            f"Warning: Model weights not found at {model_weights_path}. Using untrained model."
        )

    model.eval()

    # 3. Run Inference
    all_residuals = []
    all_baselines = []

    print(f"Starting inference on {len(test_dataset)} samples...")
    with torch.no_grad():
        for features, wls_baseline in test_loader:
            features = features.to(device)

            # Forward pass
            # Model outputs residuals: (Batch, 2) -> [Lat_Residual, Lon_Residual]
            residuals = model(features).cpu().numpy()

            all_residuals.append(residuals)
            all_baselines.append(wls_baseline.numpy())

    if not all_residuals:
        print("No predictions generated. Check test metadata and data availability.")
        return

    # Concatenate results from all batches
    residuals = np.concatenate(all_residuals, axis=0)
    baselines = np.concatenate(all_baselines, axis=0)

    # 4. Reconstruct Absolute Coordinates
    # Prediction = Baseline + Residual
    # baselines[:, 0] is WlsLat, baselines[:, 1] is WlsLon
    pred_lat = baselines[:, 0] + residuals[:, 0]
    pred_lon = baselines[:, 1] + residuals[:, 1]

    # 5. Construct Submission DataFrame
    # We retrieve the original metadata rows corresponding to the valid prediction windows.
    # SmartphoneDataset.indices stores the indices of the full_df that were valid targets.
    valid_indices = test_dataset.indices

    # Extract the relevant rows from the full dataframe used by the dataset
    submission_df = test_dataset.full_df.iloc[valid_indices].copy()

    # Assign predicted values
    submission_df["LatitudeDegrees"] = pred_lat
    submission_df["LongitudeDegrees"] = pred_lon

    # Rename utcTimeMillis to UnixTimeMillis as required by submission format
    submission_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    # Select and order columns as per sample_submission.csv
    final_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    final_submission = submission_df[final_cols]

    # 6. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(final_submission)} rows.")
