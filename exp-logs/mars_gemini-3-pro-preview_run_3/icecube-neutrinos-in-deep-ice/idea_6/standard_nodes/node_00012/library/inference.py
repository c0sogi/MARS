import os
import time
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library import utils, data
from library.model import CFDGN


def predict_and_format(model, loader, device):
    """
    Generates predictions for the test set, handling the inverse rotation
    from Canonical Frame to Global Frame.

    Args:
        model: Trained CFDGN model.
        loader: DataLoader for the test set.
        device: Torch device.

    Returns:
        pd.DataFrame: Submission dataframe with event_id, azimuth, zenith.
    """
    model.eval()
    results = []

    # Ensure no gradients are computed
    with torch.no_grad():
        for batch in loader:
            # Move data to device
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)

            # Rotation matrices and event IDs stay on CPU/Numpy for post-processing
            # rotation shape: (Batch, 3, 3)
            rotation = batch["rotation"].numpy()
            event_ids = batch["event_id"].numpy()

            # Predict direction in Canonical Frame: (Batch, 3)
            # The model outputs a normalized unit vector in the local frame
            pred_local = model(x, mask).cpu().numpy()

            # Inverse Rotation: Transform back to Global Frame
            # The rotation matrix R aligns the Global Frame to the Canonical Frame (Local).
            # Relationship: v_local = v_global @ R.T (assuming row vectors)
            # Inverse: v_global = v_local @ R
            #
            # We use einsum for batch matrix multiplication:
            # "bji,bj->bi"
            # b: batch dimension
            # j: local dimension (summed over)
            # i: global dimension (output)
            # This effectively performs v_global[b, i] = sum_j (Rotation[b, j, i] * v_local[b, j])
            pred_global = np.einsum("bji,bj->bi", rotation, pred_local)

            # Convert Cartesian to Spherical (Azimuth, Zenith)
            az, ze = utils.cartesian_to_spherical(
                pred_global[:, 0], pred_global[:, 1], pred_global[:, 2]
            )

            # Collect results
            for eid, a, z in zip(event_ids, az, ze):
                results.append({"event_id": int(eid), "azimuth": a, "zenith": z})

    # Create DataFrame
    df_sub = pd.DataFrame(results)

    # Ensure correct column order
    df_sub = df_sub[["event_id", "azimuth", "zenith"]]

    return df_sub


def run_inference(model_path=None, output_path=None, device=None):
    """
    Main inference pipeline.

    Args:
        model_path (str, optional): Path to the trained model weights.
        output_path (str, optional): Path to save the submission CSV.
        device (str, optional): Device to run inference on ('cuda' or 'cpu').
    """
    # Setup directories
    Config.setup_directories()

    # Set defaults if not provided
    if model_path is None:
        model_path = Config.MODEL_PATH
    if output_path is None:
        output_path = Config.SUBMISSION_PATH
    if device is None:
        device = Config.DEVICE

    device = torch.device(device)
    print(f"Initializing Inference on {device}...")

    # Load Data
    # get_dataloaders returns (train, val, test). We only need test.
    print("Loading test data...")
    _, _, test_loader = data.get_dataloaders()

    # Initialize Model
    model = CFDGN().to(device)

    # Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate Predictions
    print("Generating predictions...")
    start_time = time.time()
    df_submission = predict_and_format(model, test_loader, device)
    duration = time.time() - start_time
    print(f"Inference complete in {duration:.2f} seconds.")

    # Save Submission
    print(f"Saving submission to {output_path}...")
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved with {len(df_submission)} rows.")
