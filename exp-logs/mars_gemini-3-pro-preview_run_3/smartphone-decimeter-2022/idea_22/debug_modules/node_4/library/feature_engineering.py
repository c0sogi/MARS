import numpy as np
import pandas as pd
import os
from library.config import L1_SIGNALS, L5_SIGNALS, WORKING_DIR, LIGHT_SPEED
from library.utils import ecef_to_wgs84, ecef_to_enu, process_with_cache
from library.data_loader import load_dataset


class SplitBandFeatureExtractor:
    def __init__(self):
        self.l1_signals = L1_SIGNALS
        self.l5_signals = L5_SIGNALS

    def compute_los_vectors(self, df):
        """
        Computes Line-of-Sight (LOS) vectors from WLS user position to Satellite in ENU frame.
        Adds 'los_e', 'los_n', 'los_u', and 'sv_dist' columns to df.
        """
        # Convert WLS ECEF to LLA for ENU projection origin
        # We assume WlsPosition is present
        wls_x = df["WlsPositionXEcefMeters"].values
        wls_y = df["WlsPositionYEcefMeters"].values
        wls_z = df["WlsPositionZEcefMeters"].values

        # Vectorized conversion to get local tangent plane origin
        lat, lon, alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

        # Satellite Positions
        sv_x = df["SvPositionXEcefMeters"].values
        sv_y = df["SvPositionYEcefMeters"].values
        sv_z = df["SvPositionZEcefMeters"].values

        # ENU Vector (Unnormalized)
        # ecef_to_enu calculates vector from origin (lat, lon, alt) to target (sv_x, sv_y, sv_z)
        e, n, u = ecef_to_enu(sv_x, sv_y, sv_z, lat, lon, alt)

        # Calculate Distance
        dist = np.sqrt(e**2 + n**2 + u**2)
        # Avoid division by zero
        dist = np.where(dist < 1e-3, 1e-3, dist)

        df["los_e"] = e / dist
        df["los_n"] = n / dist
        df["los_u"] = u / dist
        df["sv_dist"] = dist

        return df

    def calculate_residuals(self, df):
        """
        Computes Pseudorange and Doppler residuals.
        Adds 'pr_residual' and 'dopp_residual' columns to df.
        """
        # --- Pseudorange Residuals ---
        # Fill NaNs with 0 for corrections to avoid losing data
        df["SvClockBiasMeters"] = df["SvClockBiasMeters"].fillna(0)
        df["IsrbMeters"] = df["IsrbMeters"].fillna(0)
        df["IonosphericDelayMeters"] = df["IonosphericDelayMeters"].fillna(0)
        df["TroposphericDelayMeters"] = df["TroposphericDelayMeters"].fillna(0)

        # Corrected Pseudorange: Raw + SatClock - Isrb - Iono - Tropo
        corrected_pr = (
            df["RawPseudorangeMeters"]
            + df["SvClockBiasMeters"]
            - df["IsrbMeters"]
            - df["IonosphericDelayMeters"]
            - df["TroposphericDelayMeters"]
        )

        # Geometric Distance (computed in compute_los_vectors)
        if "sv_dist" not in df.columns:
            raise ValueError("LOS vectors must be computed before residuals.")

        # Raw Residual (contains Receiver Clock Bias + Position Error + Noise)
        # Residual = Measured - Expected
        raw_res = corrected_pr - df["sv_dist"]

        # Remove Receiver Clock Bias (Common Mode Error)
        # We estimate it as the median residual per epoch to be robust against outliers
        df["raw_res_temp"] = raw_res
        clock_bias_est = df.groupby(["tripId", "UnixTimeMillis"])[
            "raw_res_temp"
        ].transform("median")
        df["pr_residual"] = raw_res - clock_bias_est

        # --- Doppler Residuals ---
        # Measured Rate
        meas_rate = df["PseudorangeRateMetersPerSecond"]

        # Expected Rate: SvVelocity projected onto LOS
        # SvVelocity is in ECEF. Project onto LOS vector in ECEF.
        dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
        dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
        dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]
        dist = df["sv_dist"]

        # Unit vectors in ECEF
        u_x = dx / dist
        u_y = dy / dist
        u_z = dz / dist

        # Dot product with SvVelocity
        sv_vel_proj = (
            df["SvVelocityXEcefMetersPerSecond"] * u_x
            + df["SvVelocityYEcefMetersPerSecond"] * u_y
            + df["SvVelocityZEcefMetersPerSecond"] * u_z
        )

        # Raw Doppler Residual
        # Note: If sat moves away, dist increases, rate is positive.
        dopp_res = meas_rate - sv_vel_proj

        # Remove Clock Drift (Common Mode)
        df["dopp_res_temp"] = dopp_res
        drift_est = df.groupby(["tripId", "UnixTimeMillis"])["dopp_res_temp"].transform(
            "median"
        )
        df["dopp_residual"] = dopp_res - drift_est

        # Cleanup temps
        df.drop(columns=["raw_res_temp", "dopp_res_temp"], inplace=True)

        return df

    def aggregate_forces(self, df):
        """
        Aggregates residuals into force vectors split by band (L1/L5).
        Returns a DataFrame aggregated by epoch.
        """
        # Define weights using Inverse Variance Weighting
        # Fill NaN uncertainty with a high value (low weight)
        sigma = df["RawPseudorangeUncertaintyMeters"].fillna(100.0)
        df["weight"] = 1.0 / (sigma**2 + 1e-6)

        # Identify Bands
        df["is_l5"] = df["SignalType"].isin(self.l5_signals)
        df["is_l1"] = df["SignalType"].isin(self.l1_signals)

        # Compute Force Components per signal
        # Force = weight * residual * unit_vector
        df["f_pr_e"] = df["weight"] * df["pr_residual"] * df["los_e"]
        df["f_pr_n"] = df["weight"] * df["pr_residual"] * df["los_n"]

        # Doppler Weights
        sigma_rate = df["PseudorangeRateUncertaintyMetersPerSecond"].fillna(10.0)
        w_rate = 1.0 / (sigma_rate**2 + 1e-6)
        df["f_dopp_e"] = w_rate * df["dopp_residual"] * df["los_e"]
        df["f_dopp_n"] = w_rate * df["dopp_residual"] * df["los_n"]

        # Create masked columns for aggregation
        for band, mask_col in [("L1", "is_l1"), ("L5", "is_l5")]:
            mask = df[mask_col]
            df[f"{band}_w_sum"] = np.where(mask, df["weight"], 0)
            df[f"{band}_E_force"] = np.where(mask, df["f_pr_e"], 0)
            df[f"{band}_N_force"] = np.where(mask, df["f_pr_n"], 0)

            df[f"{band}_w_rate_sum"] = np.where(mask, w_rate, 0)
            df[f"{band}_E_vel_force"] = np.where(mask, df["f_dopp_e"], 0)
            df[f"{band}_N_vel_force"] = np.where(mask, df["f_dopp_n"], 0)

        # Group by Epoch
        agg_dict = {
            "L1_w_sum": "sum",
            "L1_E_force": "sum",
            "L1_N_force": "sum",
            "L5_w_sum": "sum",
            "L5_E_force": "sum",
            "L5_N_force": "sum",
            "L1_w_rate_sum": "sum",
            "L1_E_vel_force": "sum",
            "L1_N_vel_force": "sum",
            "L5_w_rate_sum": "sum",
            "L5_E_vel_force": "sum",
            "L5_N_vel_force": "sum",
            "Svid": "count",  # Sat count
        }

        # Keep WLS position for the anchor
        first_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        for c in first_cols:
            if c in df.columns:
                agg_dict[c] = "first"

        grouped = df.groupby(["tripId", "UnixTimeMillis"]).agg(agg_dict).reset_index()

        # Normalize forces by total weight (Average Residual Vector)
        for band in ["L1", "L5"]:
            w = grouped[f"{band}_w_sum"] + 1e-9
            grouped[f"{band}_E_force"] /= w
            grouped[f"{band}_N_force"] /= w

            w_rate = grouped[f"{band}_w_rate_sum"] + 1e-9
            grouped[f"{band}_E_vel_force"] /= w_rate
            grouped[f"{band}_N_vel_force"] /= w_rate

        return grouped


def _process_dataset_chunk(df_chunk):
    """
    Helper to process a chunk of the dataset.
    """
    extractor = SplitBandFeatureExtractor()

    # 1. Compute LOS
    df_chunk = extractor.compute_los_vectors(df_chunk)

    # 2. Compute Residuals
    df_chunk = extractor.calculate_residuals(df_chunk)

    # 3. Aggregate
    df_features = extractor.aggregate_forces(df_chunk)

    return df_features


def extract_features(split, load_cached_data=True, max_drives=None):
    """
    Main entry point for feature extraction.
    Loads raw data, computes features, and caches the result.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached parquet files.
        max_drives (int): Limit number of drives for debugging.

    Returns:
        pd.DataFrame: Aggregated features indexed by tripId and timestamp.
    """

    def _compute():
        # Load raw data (merged GNSS+IMU+GT)
        print(f"Loading raw dataset for split: {split}")
        df_raw = load_dataset(
            split, load_cached_data=load_cached_data, max_drives=max_drives
        )

        if df_raw.empty:
            print("Warning: Raw dataset is empty.")
            return pd.DataFrame()

        print(f"Computing features for {len(df_raw)} measurements...")
        df_feats = _process_dataset_chunk(df_raw)

        # If train/val, we need targets attached to the aggregated rows.
        # The raw data had GT columns repeated per satellite; take the first one per epoch.
        if "LatitudeDegrees" in df_raw.columns:
            gt_cols = ["LatitudeDegrees", "LongitudeDegrees", "AltitudeMeters"]
            # Ensure columns exist
            cols_to_fetch = [c for c in gt_cols if c in df_raw.columns]
            if cols_to_fetch:
                df_gt = (
                    df_raw.groupby(["tripId", "UnixTimeMillis"])[cols_to_fetch]
                    .first()
                    .reset_index()
                )
                df_feats = pd.merge(
                    df_feats, df_gt, on=["tripId", "UnixTimeMillis"], how="left"
                )

        return df_feats

    # Construct cache name
    suffix = f"_{max_drives}" if max_drives else ""
    cache_name = f"features_{split}{suffix}.parquet"

    return process_with_cache(cache_name, _compute, load_cached_data=load_cached_data)
