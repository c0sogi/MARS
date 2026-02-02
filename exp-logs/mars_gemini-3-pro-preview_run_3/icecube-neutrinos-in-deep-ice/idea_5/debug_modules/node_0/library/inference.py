import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.network import ADGN_Model
from library.data_loader import IceCubeDataset
from library.utils import cartesian_to_spherical


def predict(model, loader, device):
    """
    Runs inference on the provided loader using the model.

    Args:
        model (nn.Module): The trained ADGN model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        pd.DataFrame: DataFrame containing 'event_id', 'azimuth', and 'zenith'.
    """
    model.eval()
    results = []

    # Disable gradient calculation for inference to save memory and computation
    with torch.no_grad():
        for X, priors, _, event_ids in loader:
            # Move inputs to the active device
            X = X.to(device, non_blocking=True)
            priors = priors.to(device, non_blocking=True)

            # Forward pass: Model returns Cartesian unit vectors (B, 3)
            preds_cart = model(X, priors)

            # Move predictions to CPU and convert to numpy
            preds_cart = preds_cart.cpu().numpy()

            # Convert Cartesian (x, y, z) to Spherical (azimuth, zenith)
            # preds_cart[:, 0] -> x, [:, 1] -> y, [:, 2] -> z
            az, zen = cartesian_to_spherical(
                preds_cart[:, 0], preds_cart[:, 1], preds_cart[:, 2]
            )

            # Handle event_ids (convert Tensor to numpy if necessary)
            if isinstance(event_ids, torch.Tensor):
                event_ids = event_ids.numpy()

            # Create a temporary DataFrame for the current batch
            batch_df = pd.DataFrame(
                {"event_id": event_ids, "azimuth": az, "zenith": zen}
            )

            results.append(batch_df)

    # Concatenate all batch results into a single DataFrame
    if results:
        final_df = pd.concat(results, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=["event_id", "azimuth", "zenith"])

    return final_df


def generate_submission(
    model_path: str = Config.MODEL_CHECKPOINT_PATH,
    output_path: str = Config.SUBMISSION_PATH,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    limit_batches: int = None,
):
    """
    Generates the submission file for the competition.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        limit_batches (int, optional): If provided, limits the number of batches processed (for debugging).
    """
    # 1. Environment Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference device: {device}")

    # 2. Initialize Model Architecture
    print("Initializing ADGN Model architecture...")
    model = ADGN_Model().to(device)

    # 3. Load Model Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Prepare Test Dataset and Loader
    print(f"Preparing Test Dataset (limit_batches={limit_batches})...")
    # We instantiate IceCubeDataset directly to allow passing 'limit_batches'
    test_ds = IceCubeDataset(mode="test", limit_batches=limit_batches)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Test set size: {len(test_ds)} events")

    # 5. Run Inference
    print("Starting inference...")
    submission_df = predict(model, test_loader, device)

    # 6. Post-processing
    # Ensure columns are in the correct order required by the competition
    submission_df = submission_df[["event_id", "azimuth", "zenith"]]

    # Ensure event_id is strictly integer
    submission_df["event_id"] = submission_df["event_id"].astype(int)

    # 7. Save Submission
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission generation complete.")
