import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import load_sensor_geometry, direction_to_angles
from library.model import PointNetBaseline
from library.data import IceCubeBatchDataset


def predict_and_submit(config=Config):
    """
    Generates predictions for the test set and saves the submission file.
    """
    # 1. Setup
    device = torch.device(config.DEVICE)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print(f"Loading test metadata from {config.TEST_META}...")
    test_meta = pd.read_parquet(config.TEST_META)

    # Get unique batches to process
    test_batches = test_meta["batch_id"].unique()
    print(f"Found {len(test_batches)} batches in test set.")

    print(f"Loading sensor geometry from {config.SENSOR_GEOMETRY_PATH}...")
    sensor_geo = load_sensor_geometry(config.SENSOR_GEOMETRY_PATH)

    # 2. Initialize Model
    print("Initializing model...")
    model = PointNetBaseline(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    # Load Weights
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading model weights from {config.MODEL_PATH}...")
        try:
            state_dict = torch.load(
                config.MODEL_PATH, map_location=device, weights_only=True
            )
        except TypeError:
            # Fallback for older torch versions
            state_dict = torch.load(config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model file not found at {config.MODEL_PATH}. Using random weights."
        )

    model.eval()

    # 3. Inference Loop
    all_event_ids = []
    all_azimuths = []
    all_zeniths = []

    print("Starting inference...")

    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch_id in test_batches:
            # Load data for the current batch
            # IceCubeBatchDataset handles caching and preprocessing
            try:
                dataset = IceCubeBatchDataset(
                    batch_id=batch_id,
                    meta_df=test_meta,
                    sensor_geo=sensor_geo,
                    mode="test",
                    load_cached_data=True,
                )
            except Exception as e:
                print(f"Error loading batch {batch_id}: {e}")
                continue

            if len(dataset) == 0:
                continue

            # Create DataLoader
            loader = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,  # Data is already in memory
            )

            batch_preds_list = []

            for batch_data in loader:
                # dataset returns a tuple (X,) in test mode
                X = batch_data[0].to(device)

                # Forward pass
                outputs = model(X)

                # Normalize output vectors to unit length
                outputs = F.normalize(outputs, p=2, dim=1)

                batch_preds_list.append(outputs.cpu())

            # Concatenate predictions for this batch
            if len(batch_preds_list) > 0:
                batch_preds = torch.cat(batch_preds_list, dim=0)

                # Convert direction vectors to angles
                az, zen = direction_to_angles(batch_preds)

                # Store results
                all_event_ids.append(dataset.ids.numpy())
                all_azimuths.append(az.numpy())
                all_zeniths.append(zen.numpy())

            # Memory cleanup
            del dataset, loader, batch_preds_list, batch_preds, az, zen
            gc.collect()

    # 4. Aggregate and Save
    if len(all_event_ids) > 0:
        final_event_ids = np.concatenate(all_event_ids)
        final_azimuths = np.concatenate(all_azimuths)
        final_zeniths = np.concatenate(all_zeniths)

        submission_df = pd.DataFrame(
            {
                "event_id": final_event_ids,
                "azimuth": final_azimuths,
                "zenith": final_zeniths,
            }
        )

        # Sort by event_id as per submission requirements
        submission_df = submission_df.sort_values("event_id")

        print(
            f"Saving submission ({len(submission_df)} events) to {config.SUBMISSION_PATH}..."
        )
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
    else:
        print("Error: No predictions were generated.")
