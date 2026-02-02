import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    TEST_META_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    SEED,
    WORKING_DIR,
)
from library.utils import seed_everything, vector_to_angles
from library.dataset import IceCubeDataset
from library.model import GeometricPulseAggregator


def predict_test_set(
    model_path: str = os.path.join(WORKING_DIR, "best_model.pth"),
    output_path: str = SUBMISSION_PATH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    device: str = DEVICE,
    debug_subset_size: int = None,
):
    """
    Generates predictions for the test set using a trained model and saves them to a CSV file.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker processes for data loading.
        device (str): Computation device ('cpu' or 'cuda').
        debug_subset_size (int, optional): If set, limits the number of test events for debugging.
    """
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Starting inference on device: {device}")
    print(f"Loading model from: {model_path}")

    # 2. Load Test Data
    # We use the existing metadata file for the test set
    test_dataset = IceCubeDataset(
        metadata_path=TEST_META_PATH, mode="test", debug_subset_size=debug_subset_size
    )

    # Important: shuffle=False to ensure we match predictions to event_ids correctly
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"Test dataset loaded. Total events: {len(test_dataset)}")

    # 3. Initialize Model and Load Weights
    model = GeometricPulseAggregator().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_azimuths = []
    all_zeniths = []
    all_event_ids = []

    # We need to access the dataframe directly to get event_ids efficiently
    # The loader is sequential, so we can slice the metadata dataframe
    meta_df = test_dataset.meta_df
    current_idx = 0

    print("Running inference...")

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)

            # Forward pass
            outputs = model(features)

            # Convert 3D vectors to Azimuth and Zenith
            # vector_to_angles returns tensors on the same device
            az_tensor, ze_tensor = vector_to_angles(outputs)

            # Move to CPU and convert to numpy
            batch_azimuths = az_tensor.cpu().numpy()
            batch_zeniths = ze_tensor.cpu().numpy()

            all_azimuths.append(batch_azimuths)
            all_zeniths.append(batch_zeniths)

            # Get corresponding Event IDs
            batch_len = features.size(0)
            batch_event_ids = meta_df.iloc[current_idx : current_idx + batch_len][
                "event_id"
            ].values
            all_event_ids.append(batch_event_ids)

            current_idx += batch_len

    # 5. Aggregate Results
    print("Aggregating results...")
    final_azimuths = np.concatenate(all_azimuths)
    final_zeniths = np.concatenate(all_zeniths)
    final_event_ids = np.concatenate(all_event_ids)

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "event_id": final_event_ids,
            "azimuth": final_azimuths,
            "zenith": final_zeniths,
        }
    )

    # 7. Save to CSV
    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission generation complete.")
