import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.data import IceCubeDataset, process_batch
from library.model import DV_AGN
from library.utils import seed_everything, direction_to_angles
from library.geometry import load_sensor_geometry


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test set using the trained DV-AGN model.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed batch data from cache.
                                 If False or if cache is missing, data is processed from scratch.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = Config.DEVICE

    model_path = os.path.join(Config.WORKING_DIR, "model.pth")
    if not os.path.exists(model_path):
        print(
            f"Error: Trained model not found at {model_path}. Cannot generate submission."
        )
        return

    print(f"Inference Device: {device}")

    # 2. Load Resources
    print("Loading sensor geometry...")
    sensor_map = load_sensor_geometry(Config.SENSOR_GEO_PATH)

    print("Loading test metadata...")
    test_meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.parquet")
    if not os.path.exists(test_meta_path):
        print(f"Error: Test metadata not found at {test_meta_path}")
        return

    test_meta = pd.read_parquet(test_meta_path)
    test_batches = test_meta["batch_id"].unique()

    # Sort batches to ensure consistent processing order (though not strictly required for output)
    test_batches.sort()

    # 3. Initialize Model
    print(f"Loading model from {model_path}...")
    model = DV_AGN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_ids = []
    all_azimuth = []
    all_zenith = []

    print(f"Starting inference on {len(test_batches)} batches...")

    with torch.no_grad():
        for i, batch_id in enumerate(test_batches):
            # Process batch (handles caching internally)
            # For test mode, process_batch returns ids as the third element
            X_raw, X_canon, batch_event_ids = process_batch(
                batch_id,
                test_meta,
                sensor_map,
                mode="test",
                load_cached_data=load_cached_data,
            )

            # Create Dataset and Loader
            # We pass batch_event_ids as 'y' so the loader yields them alongside features
            dataset = IceCubeDataset(X_raw, X_canon, batch_event_ids)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            for xr, xc, ids in loader:
                xr = xr.to(device)
                xc = xc.to(device)

                # Forward pass: Model outputs 3D direction vectors
                pred_vectors = model(xr, xc)

                # Convert vectors to angles
                # direction_to_angles handles normalization internally
                az, ze = direction_to_angles(pred_vectors)

                # Collect results
                all_azimuth.extend(az.cpu().numpy())
                all_zenith.extend(ze.cpu().numpy())
                all_ids.extend(ids.numpy())

            # Memory management
            del X_raw, X_canon, batch_event_ids, dataset, loader
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # 5. Save Submission
    print("Constructing submission DataFrame...")
    submission = pd.DataFrame(
        {"event_id": all_ids, "azimuth": all_azimuth, "zenith": all_zenith}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    print(f"Saving submission to {submission_path}...")
    submission.to_csv(submission_path, index=False)

    print("Submission generation complete.")
