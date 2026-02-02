import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.model import CascadedResUNet
from library.data_loader import process_drive, GnssSequenceDataset
from library.utils import enu_to_llh


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def predict_drive(
    drive_id, phone_name, gnss_path, model, device, load_cached_data=True
):
    """
    Generates predictions for a single drive.

    Args:
        drive_id (str): The drive identifier.
        phone_name (str): The phone model name.
        gnss_path (str): Path to the GNSS log file.
        model (nn.Module): The trained PyTorch model.
        device (torch.device): Computation device.
        load_cached_data (bool): Whether to use cached processed features.

    Returns:
        pd.DataFrame: DataFrame containing 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'.
                      Returns empty DataFrame if processing fails.
    """
    # Process raw data into features
    df = process_drive(
        drive_id,
        phone_name,
        gnss_path,
        gt_df=None,
        load_cached_data=load_cached_data,
    )

    if df.empty:
        return pd.DataFrame()

    # Create Dataset and Loader
    # mode='test' ensures we handle the end of the sequence with a padded/overlapping window
    dataset = GnssSequenceDataset(
        [df], sequence_length=Config.SEQUENCE_LENGTH, mode="test"
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    drive_preds = []

    model.eval()
    with torch.no_grad():
        for features, _, wls, meta in loader:
            features = features.to(device)
            wls = wls.numpy()  # Shape: (B, L, 3) -> Lat, Lon, Alt
            meta = meta.numpy()  # Shape: (B, L) -> UnixTimeMillis

            # Forward pass
            _, final_out = model(features)
            pred_enu = final_out.cpu().numpy()  # Shape: (B, 2, L) -> East, North

            # Iterate over batch
            for b in range(pred_enu.shape[0]):
                wls_b = wls[b]
                meta_b = meta[b]

                # Extract predicted residuals
                pred_e = pred_enu[b, 0, :]
                pred_n = pred_enu[b, 1, :]
                pred_u = np.zeros_like(pred_e)  # We don't predict Up

                # Extract baseline WLS coordinates
                ref_lat = wls_b[:, 0]
                ref_lon = wls_b[:, 1]
                ref_alt = wls_b[:, 2]

                # Convert ENU residuals + Baseline -> Final Lat/Lon
                pred_lat, pred_lon, _ = enu_to_llh(
                    pred_e, pred_n, pred_u, ref_lat, ref_lon, ref_alt
                )

                # Collect results for valid timestamps (ignore padding if any, though dataset handles it)
                for i in range(len(meta_b)):
                    ts = meta_b[i]
                    # Simple check to avoid padding zeros if they exist (GnssSequenceDataset pads with edge, so ts won't be 0)
                    if ts != 0:
                        drive_preds.append(
                            {
                                "UnixTimeMillis": ts,
                                "LatitudeDegrees": pred_lat[i],
                                "LongitudeDegrees": pred_lon[i],
                            }
                        )

    if not drive_preds:
        return pd.DataFrame()

    # Create DataFrame
    pred_df = pd.DataFrame(drive_preds)

    # Aggregate predictions for overlapping timestamps (due to sliding window stride)
    # We take the mean of predictions for the same timestamp
    pred_df = pred_df.groupby("UnixTimeMillis", as_index=False).mean()

    return pred_df


def generate_submission(load_cached_data=True):
    """
    Main inference function to generate the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from cache.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    weights_path = os.path.join("./working", "model_weights.pth")
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return

    model = CascadedResUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Model loaded from {weights_path}")

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Error: Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Identify unique drives to process
    # We group by drive_id and phone_name to process each phone's trip sequentially
    unique_drives = test_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    print(f"Processing {len(unique_drives)} unique test drives...")

    results = []

    for _, row in unique_drives.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        # Predict for this drive
        drive_pred_df = predict_drive(
            drive_id, phone_name, gnss_path, model, device, load_cached_data
        )

        if drive_pred_df.empty:
            print(f"Warning: No predictions generated for {drive_id} - {phone_name}")
            continue

        # Filter test_meta for this specific drive to get the required timestamps
        drive_requirements = test_meta[
            (test_meta["drive_id"] == drive_id)
            & (test_meta["phone_name"] == phone_name)
        ].copy()

        # Align predictions with requirements
        # Feature engineering rounds timestamps to nearest second.
        # We create a JoinKey on the requirements to match the predictions.
        drive_requirements["JoinKey"] = (
            (drive_requirements["UnixTimeMillis"] + 500) // 1000 * 1000
        ).astype(np.int64)

        drive_pred_df["JoinKey"] = drive_pred_df["UnixTimeMillis"].astype(np.int64)

        # Merge
        # We drop the UnixTimeMillis from pred_df to avoid conflict, keeping the requested one
        merged = pd.merge(
            drive_requirements,
            drive_pred_df.drop(columns=["UnixTimeMillis"]),
            on="JoinKey",
            how="left",
        )

        # Interpolate to fill gaps or exact alignment issues
        cols_to_interp = ["LatitudeDegrees", "LongitudeDegrees"]
        merged[cols_to_interp] = merged[cols_to_interp].interpolate(
            method="linear", limit_direction="both"
        )

        # Keep only necessary columns for submission
        final_drive_df = merged[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]

        results.append(final_drive_df)

    # 3. Construct Final Submission
    if results:
        submission_df = pd.concat(results, ignore_index=True)

        # Load sample submission to ensure exact row order and completeness
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge our results into the sample submission structure
        final_sub = pd.merge(
            sample_sub[["tripId", "UnixTimeMillis"]],
            submission_df,
            on=["tripId", "UnixTimeMillis"],
            how="left",
        )

        # Final fallback interpolation for any remaining NaNs (e.g., if a whole drive failed)
        # We interpolate within tripId groups if possible, or global linear if desperate
        if final_sub.isnull().values.any():
            print("Warning: NaNs found in submission. Interpolating...")
            final_sub[["LatitudeDegrees", "LongitudeDegrees"]] = final_sub.groupby(
                "tripId"
            )[["LatitudeDegrees", "LongitudeDegrees"]].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
            # If still NaNs (e.g. single point trips or all empty), fill with 0 (unlikely in this dataset)
            final_sub.fillna(0, inplace=True)

        # Save
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        final_sub.to_csv(sub_path, index=False)
        print(f"Submission saved successfully to {sub_path}")
    else:
        print("Error: No results generated.")
