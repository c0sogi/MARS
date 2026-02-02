import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, WGS84Utils
from library.data_loader import load_dataset, GNSSSequenceDataset
from library.model import ResUNet1D


def generate_predictions(load_cached_data=True):
    """
    Generates predictions for the test set using the trained model.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
    """
    # Set reproducibility
    set_seed(Config.SEED)

    # 1. Load and Preprocess Test Data
    # load_dataset handles caching logic internally as per requirements
    test_df = load_dataset(
        "test", load_cached_data=load_cached_data, debug=Config.DEBUG
    )

    if test_df.empty:
        print("Test dataset is empty. Cannot generate predictions.")
        return

    # Create Dataset and DataLoader
    test_dataset = GNSSSequenceDataset(test_df, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = ResUNet1D().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print(f"Model loaded from {Config.MODEL_PATH}")
    print("Starting inference...")

    results = []

    # Metadata list from dataset matches the order of iteration
    seq_metadata = test_dataset.metadata
    seq_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            baselines = batch["baselines"].numpy()  # (B, L, 3)
            timestamps = batch["timestamps"].numpy()  # (B, L)

            # Forward pass
            # Model returns list of outputs [Head0, Head1, ...]. Head0 is full resolution.
            outputs = model(features)
            pred_res = outputs[0]  # (B, 2, L) -> (dNorth, dEast)

            # Convert to numpy (B, L, 2)
            pred_res = pred_res.permute(0, 2, 1).cpu().numpy()

            batch_size = features.shape[0]

            for b in range(batch_size):
                drive_id, phone_name = seq_metadata[seq_idx]
                seq_idx += 1

                # Determine valid sequence length from mask
                valid_len = int(mask[b].sum().item())

                # Extract valid portion of sequence
                valid_preds = pred_res[b, :valid_len, :]  # (L_valid, 2) -> (dN, dE)
                valid_base = baselines[b, :valid_len, :]  # (L_valid, 3) -> (X, Y, Z)
                valid_time = timestamps[b, :valid_len]  # (L_valid,)

                # 3. Coordinate Reconstruction
                # A. Convert Baseline WLS ECEF to Geodetic (Lat, Lon, Alt)
                wls_x = valid_base[:, 0]
                wls_y = valid_base[:, 1]
                wls_z = valid_base[:, 2]

                ref_lat, ref_lon, ref_alt = WGS84Utils.ecef_to_geodetic(
                    wls_x, wls_y, wls_z
                )

                # B. Convert Predicted ENU offsets to ECEF
                # Predictions are (dNorth, dEast). Assume dUp = 0.
                d_n = valid_preds[:, 0]
                d_e = valid_preds[:, 1]
                d_u = np.zeros_like(d_n)

                # Get corrected ECEF coordinates
                pred_x, pred_y, pred_z = WGS84Utils.enu_to_ecef(
                    d_e, d_n, d_u, ref_lat, ref_lon, ref_alt
                )

                # C. Convert Predicted ECEF back to Geodetic (Final Lat/Lon)
                final_lat, final_lon, _ = WGS84Utils.ecef_to_geodetic(
                    pred_x, pred_y, pred_z
                )

                # Create DataFrame for this sequence
                trip_id = f"{drive_id}-{phone_name}"

                seq_df = pd.DataFrame(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": valid_time,
                        "LatitudeDegrees": final_lat,
                        "LongitudeDegrees": final_lon,
                    }
                )

                results.append(seq_df)

    # 4. Generate Submission File
    if not results:
        print("No predictions generated.")
        return

    submission_df = pd.concat(results, ignore_index=True)

    # Load sample submission to ensure correct format and rows
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)

    # Ensure types match for merge
    submission_df["UnixTimeMillis"] = submission_df["UnixTimeMillis"].astype(np.int64)
    sample_sub["UnixTimeMillis"] = sample_sub["UnixTimeMillis"].astype(np.int64)

    # Merge predictions onto sample submission structure
    final_sub = pd.merge(
        sample_sub[["tripId", "UnixTimeMillis"]],
        submission_df,
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )

    # Handle missing predictions (if any)
    if final_sub.isnull().any().any():
        print(
            "Warning: Missing predictions detected. Filling with interpolation/forward-fill."
        )
        # Sort to ensure interpolation makes sense temporally
        final_sub = final_sub.sort_values(["tripId", "UnixTimeMillis"])
        # Group by trip to interpolate within trips
        final_sub = (
            final_sub.groupby("tripId")
            .apply(
                lambda group: group.interpolate(method="linear")
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
            .reset_index(drop=True)
        )

        # If still NaNs (e.g. empty trips), fill with 0 (unlikely case)
        final_sub = final_sub.fillna(0)

    # Save to disk
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
