import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import ResUNet1D
from library.data_processing import GNSSPreprocessor
from library.dataset import GnssSequenceDataset, gnss_collate_fn
from library.utils import WGS84Utils, setup_logger


class InferencePipeline:
    """
    Manages the inference process: loading data, generating predictions,
    converting coordinates, and saving the submission file.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.logger = setup_logger(os.path.join(Config.WORKING_DIR, "inference.log"))
        self.wgs84 = WGS84Utils()

    def load_model(self):
        """
        Loads the trained model architecture and weights.
        """
        model = ResUNet1D().to(self.device)

        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
            )

        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        self.logger.info(f"Model loaded from {Config.MODEL_SAVE_PATH}")
        return model

    def run(self, load_cached_data=True):
        """
        Runs the full inference pipeline.

        Args:
            load_cached_data (bool): Whether to load preprocessed data from cache.
        """
        self.logger.info("Starting inference pipeline...")

        # 1. Prepare Data
        preprocessor = GNSSPreprocessor()
        test_df = preprocessor.process_test_data(load_cached_data=load_cached_data)

        # Create Dataset and DataLoader
        test_dataset = GnssSequenceDataset(test_df, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=gnss_collate_fn,
            pin_memory=True,
        )

        self.logger.info(f"Test data loaded. Batches: {len(test_loader)}")

        # 2. Load Model
        model = self.load_model()

        # 3. Generate Predictions
        results = []

        with torch.no_grad():
            for batch in test_loader:
                # Prepare inputs: (B, T, C) -> (B, C, T)
                features = batch["features"].to(self.device).transpose(1, 2)

                # Forward pass
                # final_out: (B, 2, T) -> (Delta North, Delta East)
                final_out, _ = model(features)

                # Move to CPU for post-processing
                predictions = final_out.cpu().numpy()  # (B, 2, T)

                # Extract metadata
                drive_ids = batch["drive_id"]
                phone_names = batch["phone_name"]
                t_millis_list = batch["t_millis"]  # List of tensors
                lengths = batch["lengths"].numpy()

                # Extract WLS baselines (B, T)
                wls_lats = batch["wls_lat"].numpy()
                wls_lons = batch["wls_lon"].numpy()

                batch_size = predictions.shape[0]

                for i in range(batch_size):
                    seq_len = lengths[i]

                    # Extract valid sequence data (ignore padding)
                    # Preds shape: (2, T) -> Transpose to (T, 2) for easier handling
                    pred_seq = predictions[i, :, :seq_len].transpose()
                    d_north = pred_seq[:, 0]
                    d_east = pred_seq[:, 1]

                    ref_lat = wls_lats[i, :seq_len]
                    ref_lon = wls_lons[i, :seq_len]
                    timestamps = t_millis_list[i].numpy()

                    # Convert Cartesian residuals (Meters) to Geodetic offsets (Degrees)
                    d_lat, d_lon = self.wgs84.meters_to_degrees(
                        d_north, d_east, ref_lat
                    )

                    # Apply corrections
                    pred_lat = ref_lat + d_lat
                    pred_lon = ref_lon + d_lon

                    # Construct Trip ID
                    trip_id = f"{drive_ids[i]}-{phone_names[i]}"

                    # Store results
                    for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
                        results.append(
                            {
                                "tripId": trip_id,
                                "UnixTimeMillis": t,
                                "LatitudeDegrees": lat,
                                "LongitudeDegrees": lon,
                            }
                        )

        # 4. Create Submission DataFrame
        submission_df = pd.DataFrame(results)

        # Sort as required (usually by tripId then time)
        submission_df = submission_df.sort_values(by=["tripId", "UnixTimeMillis"])

        # 5. Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_SAVE_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")
        self.logger.info(f"Total predictions: {len(submission_df)}")
