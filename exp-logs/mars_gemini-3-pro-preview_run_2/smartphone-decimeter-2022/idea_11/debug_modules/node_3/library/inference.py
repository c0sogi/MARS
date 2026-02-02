import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.model import IMULocalTrajectoryCNN
from library.data_loader import load_data
from library.utils import meters_diff_to_latlon


def generate_submission(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, device=Config.DEVICE
):
    """
    Generates the submission file for the test set.

    Args:
        load_cached_data (bool): Whether to load pre-processed test data from cache.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cpu' or 'cuda').
    """
    print("==================================================")
    print("INFERENCE AND SUBMISSION GENERATION")
    print("==================================================")

    # 1. Load Test Data
    # load_data with mode='test' returns the dataset (X) and the metadata dataframe
    # The metadata dataframe contains 'tripId', 'UnixTimeMillis', 'WlsLat', 'WlsLon'
    # aligned with the dataset indices.
    print("Loading test data...")
    test_dataset, test_meta_df = load_data(
        mode="test", load_cached_data=load_cached_data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # 2. Load Model
    print("Loading model...")
    model = IMULocalTrajectoryCNN(
        input_dim=Config.INPUT_DIM,
        window_size=Config.WINDOW_SIZE,
        output_dim=Config.OUTPUT_DIM,
        cnn_channels=Config.CNN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        cnn_dropout=Config.CNN_DROPOUT,
        mlp_hidden_dims=Config.MLP_HIDDEN_DIMS,
        mlp_dropout=Config.MLP_DROPOUT,
    )

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train the model first."
        )

    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Inference
    print("Running inference...")
    predictions = []

    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)

            # Model outputs residuals: [DeltaEastMeters, DeltaNorthMeters]
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    # Concatenate all batch predictions
    predictions = np.concatenate(predictions, axis=0)

    # 4. Reconstruction
    print("Reconstructing absolute coordinates...")

    # Extract baseline WLS coordinates from metadata
    wls_lat = test_meta_df["WlsLat"].values
    wls_lon = test_meta_df["WlsLon"].values

    # Extract predicted residuals
    d_east = predictions[:, 0]
    d_north = predictions[:, 1]

    # Convert residuals to new Lat/Lon
    pred_lat, pred_lon = meters_diff_to_latlon(wls_lat, wls_lon, d_east, d_north)

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta_df["tripId"],
            "UnixTimeMillis": test_meta_df["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # 6. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Submission generation complete.")
    print(submission_df.head())
