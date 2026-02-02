import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, WGS84Utils
from library.dataset import load_data, GNSSDataset, gnss_collate_fn
from library.model import ResUNet1D


def generate_submission(
    load_cached_data: bool = True, batch_size: int = Config.BATCH_SIZE
):
    """
    Runs the inference pipeline: loads data, runs the model, and generates the submission file.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache if available.
        batch_size (int): Batch size for inference.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"--- Starting Inference on {device} ---")

    # 1. Load Test Data
    # The load_data function handles caching logic internally based on the flag
    print("Loading test data...")
    test_df = load_data(
        Config.TEST_METADATA_PATH, Config.TEST_CACHE, load_cached_data=load_cached_data
    )

    if test_df.empty:
        raise ValueError("Test data is empty. Cannot generate submission.")

    # 2. Prepare Dataset and Dataloader
    test_dataset = GNSSDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )
    print(f"Test Dataset: {len(test_dataset)} trips")

    # 3. Load Model
    print("Loading model...")
    model = ResUNet1D().to(device)

    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}"
        )

    checkpoint = torch.load(Config.MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Inference Loop
    results = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)  # (B, L, C)
            mask = batch["mask"].to(device)  # (B, L)
            wls_pos_list = batch["wls_pos"]  # List of tensors (L, 2)
            timestamps_list = batch["timestamps"]  # List of tensors (L,)
            trip_ids = batch["trip_ids"]  # List of strings

            # Permute features for Conv1d: (B, C, L)
            features = features.permute(0, 2, 1)

            # Forward pass
            # During eval, model returns single tensor (B, 2, L)
            outputs = model(features)

            # Permute back to (B, L, 2) for processing
            outputs = outputs.permute(0, 2, 1)

            # Process each sequence in the batch
            for i in range(len(trip_ids)):
                # Get valid length from mask
                valid_len = mask[i].sum().item()

                # Extract valid predictions and metadata
                # outputs[i] is (L_padded, 2) -> slice to (valid_len, 2)
                pred_offsets = (
                    outputs[i, :valid_len, :].cpu().numpy()
                )  # (DeltaEast, DeltaNorth)

                # Metadata was stored as lists of tensors in collate_fn, no padding applied to them
                # wls_pos_list[i] is (valid_len, 2) -> [Lat, Lon]
                base_pos = wls_pos_list[i].numpy()
                times = timestamps_list[i].numpy()
                trip_id = trip_ids[i]

                # Ensure lengths match (sanity check)
                assert (
                    len(pred_offsets) == len(base_pos) == len(times)
                ), f"Length mismatch in batch processing for {trip_id}"

                # Convert offsets (Meters) to Coordinates (Degrees)
                # pred_offsets columns: 0=East, 1=North
                delta_east = pred_offsets[:, 0]
                delta_north = pred_offsets[:, 1]

                ref_lat = base_pos[:, 0]
                ref_lon = base_pos[:, 1]

                pred_lat, pred_lon = WGS84Utils.meters_to_degrees(
                    delta_north, delta_east, ref_lat, ref_lon
                )

                # Store results
                for t, lat, lon in zip(times, pred_lat, pred_lon):
                    results.append(
                        {
                            "tripId": trip_id,
                            "UnixTimeMillis": t,
                            "LatitudeDegrees": lat,
                            "LongitudeDegrees": lon,
                        }
                    )

    # 5. Create Prediction DataFrame
    pred_df = pd.DataFrame(results)

    # 6. Merge with Sample Submission
    # The sample submission might have timestamps that are slightly offset from the rounded 1Hz
    # We match based on the rounded timestamp.

    print("Formatting submission...")
    if not os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission not found at {Config.SAMPLE_SUBMISSION_PATH}"
        )

    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Create a join key in sample submission (round to nearest second)
    sample_sub["JoinTime"] = (
        np.round(sample_sub["UnixTimeMillis"] / 1000.0) * 1000
    ).astype(np.int64)

    # Merge predictions
    # We use left join to preserve the structure of sample_submission
    submission = pd.merge(
        sample_sub,
        pred_df,
        left_on=["tripId", "JoinTime"],
        right_on=["tripId", "UnixTimeMillis"],
        how="left",
        suffixes=("_old", ""),
    )

    # Fill missing predictions with original values (fallback) or interpolate
    # Here we use linear interpolation for small gaps if possible, or forward fill
    # However, since we don't have the original WLS in sample_sub (it just has dummy values usually),
    # we should check if we have missing values.

    missing_mask = submission["LatitudeDegrees"].isna()
    if missing_mask.sum() > 0:
        print(
            f"Warning: {missing_mask.sum()} timestamps in submission did not get a prediction."
        )
        # Fallback strategy: Interpolate within trip
        # Sort by trip and time
        submission = submission.sort_values(["tripId", "UnixTimeMillis_old"])

        # Interpolate Lat/Lon
        submission["LatitudeDegrees"] = submission.groupby("tripId")[
            "LatitudeDegrees"
        ].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
        submission["LongitudeDegrees"] = submission.groupby("tripId")[
            "LongitudeDegrees"
        ].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))

        # If still NaN (e.g. whole trip missing), this is critical.
        # In this specific challenge context, we expect coverage.
        # If completely missing, we can't do much without the WLS baseline from raw files for those specific timestamps.
        # Assuming test_processed.parquet covers all drives in sample_submission.

    # Select final columns
    final_submission = submission[
        ["tripId", "UnixTimeMillis_old", "LatitudeDegrees", "LongitudeDegrees"]
    ].rename(columns={"UnixTimeMillis_old": "UnixTimeMillis"})

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    final_submission.to_csv(Config.SUBMISSION_OUTPUT, index=False)
    print(f"Submission saved to {Config.SUBMISSION_OUTPUT}")
    print(f"Total rows: {len(final_submission)}")
