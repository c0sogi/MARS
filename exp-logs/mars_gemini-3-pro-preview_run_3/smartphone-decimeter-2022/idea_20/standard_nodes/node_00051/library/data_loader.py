import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from library.utils import IOHelper, CoordinateTransformer


class GnssDataset:
    def __init__(self, mode="train", root_dir="./input"):
        self.mode = mode
        self.root_dir = root_dir
        self.metadata_path = f"./metadata/{mode}_metadata.csv"

    def _load_imu(self, imu_path):
        """
        Loads and processes IMU data.
        Pivots the data to have columns for Accel, Gyro, Mag aligned by timestamp.
        """
        try:
            df_imu = pd.read_csv(imu_path)
            # Pivot to get MeasurementX, Y, Z for each MessageType
            # We take the mean if there are duplicates for the exact same millisecond

            # Filter relevant columns
            df_imu = df_imu[
                [
                    "utcTimeMillis",
                    "MessageType",
                    "MeasurementX",
                    "MeasurementY",
                    "MeasurementZ",
                ]
            ]

            # Group by timestamp and message type, taking mean to handle duplicates
            df_imu = (
                df_imu.groupby(["utcTimeMillis", "MessageType"]).mean().reset_index()
            )

            # Pivot
            df_imu_pivoted = df_imu.pivot(
                index="utcTimeMillis",
                columns="MessageType",
                values=["MeasurementX", "MeasurementY", "MeasurementZ"],
            )

            # Flatten columns
            # Resulting cols example: UncalAccel_MeasurementX, UncalGyro_MeasurementZ
            df_imu_pivoted.columns = [
                f"{col[1]}_{col[0]}" for col in df_imu_pivoted.columns
            ]
            df_imu_pivoted.reset_index(inplace=True)

            return df_imu_pivoted
        except Exception as e:
            print(f"Warning: Failed to load IMU data from {imu_path}: {e}")
            return pd.DataFrame(columns=["utcTimeMillis"])

    def _process_trip(self, trip_id, gnss_rel_path, imu_rel_path, gt_rel_path=None):
        """
        Loads GNSS, IMU, and GT for a single trip and merges them.
        """
        gnss_path = os.path.join(self.root_dir, gnss_rel_path)
        imu_path = os.path.join(self.root_dir, imu_rel_path)

        # 1. Load GNSS
        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found: {gnss_path}")
            return pd.DataFrame()

        df_gnss = pd.read_csv(gnss_path)

        # 2. Load IMU and merge
        if os.path.exists(imu_path):
            df_imu = self._load_imu(imu_path)

            if not df_imu.empty:
                # GNSS timestamps are utcTimeMillis. IMU are also utcTimeMillis.
                df_gnss = df_gnss.sort_values("utcTimeMillis")
                df_imu = df_imu.sort_values("utcTimeMillis")

                # Merge nearest IMU data to each GNSS epoch
                df_gnss = pd.merge_asof(
                    df_gnss,
                    df_imu,
                    on="utcTimeMillis",
                    direction="nearest",
                    tolerance=1000,  # 1 second tolerance
                )

        # 3. Load Ground Truth if available (Train/Val)
        if self.mode in ["train", "val"] and gt_rel_path:
            gt_path_abs = os.path.join(self.root_dir, gt_rel_path)
            if os.path.exists(gt_path_abs):
                df_gt = pd.read_csv(gt_path_abs)

                # GT uses UnixTimeMillis, GNSS uses utcTimeMillis. They are the same scale.
                # Select relevant GT columns
                gt_cols = [
                    "UnixTimeMillis",
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                    "AltitudeMeters",
                    "SpeedMps",
                    "AccuracyMeters",
                    "BearingDegrees",
                ]
                gt_cols = [c for c in gt_cols if c in df_gt.columns]
                df_gt = df_gt[gt_cols]

                # Rename for merge
                df_gt = df_gt.rename(columns={"UnixTimeMillis": "utcTimeMillis"})

                # Inner merge ensures we only keep GNSS epochs that have a Ground Truth label
                # This effectively filters out non-labeled epochs in training
                df_merged = pd.merge(df_gnss, df_gt, on="utcTimeMillis", how="inner")
            else:
                print(f"Warning: GT file missing {gt_path_abs}")
                return pd.DataFrame()

        else:
            # Test mode: Keep all GNSS rows.
            df_merged = df_gnss

        df_merged["tripId"] = trip_id
        return df_merged

    def load(self, load_cached_data=True):
        """
        Main method to load the dataset.
        """
        cache_file = f"dataset_{self.mode}.parquet"

        # 1. Try Cache
        if load_cached_data:
            df = IOHelper.load_parquet(cache_file)
            if df is not None:
                return df

        # 2. Process from scratch
        print(f"Processing {self.mode} dataset from raw files...")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df_meta = pd.read_csv(self.metadata_path)

        # Identify unique trips
        cols_to_keep = ["tripId", "drive_id", "phone_name", "gnss_path", "imu_path"]
        if "gt_path" in df_meta.columns:
            cols_to_keep.append("gt_path")

        trips = df_meta[cols_to_keep].drop_duplicates()

        results = []

        # Iterate over trips
        for _, row in tqdm(
            trips.iterrows(), total=len(trips), desc=f"Loading {self.mode} trips"
        ):
            trip_id = row["tripId"]
            gnss_path = row["gnss_path"]
            imu_path = row["imu_path"]
            gt_path = row["gt_path"] if "gt_path" in row else None

            try:
                df_trip = self._process_trip(trip_id, gnss_path, imu_path, gt_path)
                if not df_trip.empty:
                    df_trip["drive_id"] = row["drive_id"]
                    df_trip["phone_name"] = row["phone_name"]
                    results.append(df_trip)
            except Exception as e:
                print(f"Error processing trip {trip_id}: {e}")

        if not results:
            raise ValueError("No data loaded!")

        final_df = pd.concat(results, ignore_index=True)

        # 3. Save to Cache
        IOHelper.save_parquet(final_df, cache_file)

        return final_df


def load_data(mode="train", load_cached=True):
    """
    Wrapper function to initialize loader and return data.
    """
    loader = GnssDataset(mode=mode)
    return loader.load(load_cached_data=load_cached)
