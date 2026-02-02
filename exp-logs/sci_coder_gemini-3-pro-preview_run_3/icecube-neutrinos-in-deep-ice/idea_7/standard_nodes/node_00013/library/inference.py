import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from library.config import Config
from library.dataset import IceCubeDataset
from library.model import DFCGN
from library.utils import direction_to_angles


def predict_and_submit(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, batch_ids=None
):
    """
    Generates predictions for the test set using the trained DF-CGN model and saves the submission file.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        batch_ids (list, optional): List of batch IDs to process. Useful for debugging or partial inference.
                                    If None, processes all batches defined in test metadata.
    """
    # 1. Setup Environment
    device = torch.device(Config.DEVICE)
    print(f"Inference using device: {device}")

    # Ensure output directory exists
    # Config.SUBMISSION_PATH is a Path object, e.g., working/idea_7/submission.csv
    submission_dir = Config.SUBMISSION_PATH.parent
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Initialize Dataset and DataLoader
    print("Initializing Test Dataset...")
    # IceCubeDataset handles metadata loading, caching of processed .npy files,
    # and SVD/Canonical frame transformation internally.
    test_dataset = IceCubeDataset(mode="test", batch_ids=batch_ids)

    print(f"Total test samples to process: {len(test_dataset)}")

    # We use shuffle=False to ensure that the order of predictions matches
    # the order of event_ids in test_dataset.metadata.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Load Model
    print("Loading Model...")
    model = DFCGN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Please train the model first."
        )

    # Load weights
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    print("Starting Inference...")
    all_azimuth = []
    all_zenith = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass: Model returns direction vectors (Batch, 3)
            # The model handles the dual-frame input (raw + canonical) internally via the dataset features
            pred_vec = model(batch)

            # Convert direction vectors to angles (azimuth, zenith)
            # direction_to_angles handles normalization and clamping
            az, zen = direction_to_angles(pred_vec)

            # Move results to CPU and convert to numpy
            all_azimuth.append(az.cpu().numpy())
            all_zenith.append(zen.cpu().numpy())

    # Concatenate predictions from all batches
    if len(all_azimuth) > 0:
        final_azimuth = np.concatenate(all_azimuth)
        final_zenith = np.concatenate(all_zenith)
    else:
        final_azimuth = np.array([])
        final_zenith = np.array([])

    # 5. Construct Submission DataFrame
    print("Constructing Submission DataFrame...")

    # Retrieve event_ids from the dataset metadata.
    # The dataset sorts metadata by [batch_id, event_id] in __init__,
    # and DataLoader(shuffle=False) respects this order.
    event_ids = test_dataset.metadata["event_id"].values

    # Validation check
    if len(event_ids) != len(final_azimuth):
        raise ValueError(
            f"Mismatch between number of predictions ({len(final_azimuth)}) "
            f"and number of events in metadata ({len(event_ids)})."
        )

    df_submission = pd.DataFrame(
        {"event_id": event_ids, "azimuth": final_azimuth, "zenith": final_zenith}
    )

    # 6. Save to CSV
    output_path = str(Config.SUBMISSION_PATH)
    print(f"Saving submission to {output_path}...")

    # Ensure float precision is preserved
    df_submission.to_csv(output_path, index=False, float_format="%.6f")

    print("Submission saved successfully.")
