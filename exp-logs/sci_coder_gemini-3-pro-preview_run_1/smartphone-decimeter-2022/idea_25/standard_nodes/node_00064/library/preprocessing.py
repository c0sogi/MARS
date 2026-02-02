import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import wgs84_to_enu, ecef_to_lla


class GNSSPreprocessor:
    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.input_dir = Config.INPUT_DIR

    def _get_cache_path(self, split):
        return os.path.join(self.cache_dir, f"{split}_processed.parquet")

    def _process_drive(self, drive_id, phone_name, df_meta_drive, is_train):
        # Construct path
        # Correction: The input folder structure is train/ or test/.
        # df_meta_drive comes from train_metadata or test_metadata.
        # We need to determine the source folder based on the split passed to process_data,
        # but here we rely on the caller or metadata.
        # Actually, train and val splits both come from the 'train' folder in input.
        # Test split comes from 'test' folder.
        # We can infer from the file path in metadata if available, but let's use the is_train flag logic carefully.
        # A safer way is to check if the drive exists in train or test.

        path_train = os.path.join(
            self.input_dir, "train", drive_id, phone_name, "device_gnss.csv"
        )
        path_test = os.path.join(
            self.input_dir, "test", drive_id, phone_name, "device_gnss.csv"
        )

        if os.path.exists(path_train):
            gnss_path = path_train
        elif os.path.exists(path_test):
            gnss_path = path_test
        else:
            # Fallback or error
            return pd.DataFrame()

        # Load GNSS data
        use_cols = [
            "utcTimeMillis",
            "SignalType",
            "SvElevationDegrees",
            "SvAzimuthDegrees",
            "Cn0DbHz",
            "RawPseudorangeUncertaintyMeters",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "AccumulatedDeltaRangeState",
        ]

        try:
            df_gnss = pd.read_csv(gnss_path, usecols=lambda c: c in use_cols)
        except ValueError:
            # Fallback if some columns missing (should not happen based on dataset desc, but safe)
            df_gnss = pd.read_csv(gnss_path)

        # Filter GNSS to relevant timestamps in metadata
        valid_timestamps = df_meta_drive["UnixTimeMillis"].unique()
        df_gnss = df_gnss[df_gnss["utcTimeMillis"].isin(valid_timestamps)].copy()

        if df_gnss.empty:
            return pd.DataFrame()

        # --- Stratification ---
        # Stratum 2: High Precision
        # L5 signals: GPS_L5, GAL_E5A, BDS_B2A, QZS_J5
        l5_signals = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"]
        # Valid Carrier Phase: AccumulatedDeltaRangeState bit 0 (1) is set.
        adr_state = df_gnss["AccumulatedDeltaRangeState"].fillna(0).astype(int)
        has_phase = (adr_state & 1) == 1
        is_l5 = df_gnss["SignalType"].isin(l5_signals)
        mask_hp = is_l5 | has_phase

        # Stratum 3: High Risk (Elevation < 30)
        mask_hr = df_gnss["SvElevationDegrees"] < 30

        # --- Aggregation ---
        def agg_stratum(mask, prefix):
            subset = df_gnss[mask]
            if subset.empty:
                return None

            grouped = subset.groupby("utcTimeMillis")
            agg_dict = {}
            for col in Config.STRATUM_RAW_FIELDS:  # Cn0DbHz, SvElevationDegrees
                for stat in Config.STRATUM_STATS:  # mean, std, min, max
                    agg_dict[f"{prefix}_{col}_{stat}"] = pd.NamedAgg(
                        column=col, aggfunc=stat
                    )

            return grouped.agg(**agg_dict)

        # 1. Global
        feat_global = agg_stratum(slice(None), "global")

        # 2. High Precision
        feat_hp = agg_stratum(mask_hp, "high_precision")

        # 3. High Risk
        feat_hr = agg_stratum(mask_hr, "high_risk")

        # --- Global Context Features ---
        grouped_all = df_gnss.groupby("utcTimeMillis")
        ctx_features = grouped_all.agg(
            SignalCount=pd.NamedAgg(column="Cn0DbHz", aggfunc="count"),
            RawPseudorangeUncertaintyMeters_mean=pd.NamedAgg(
                column="RawPseudorangeUncertaintyMeters", aggfunc="mean"
            ),
        )

        # Azimuth Centroid
        az_rad = np.radians(df_gnss["SvAzimuthDegrees"].fillna(0))
        weights = 10 ** (df_gnss["Cn0DbHz"].fillna(0) / 10.0)
        df_gnss["_w_sin"] = weights * np.sin(az_rad)
        df_gnss["_w_cos"] = weights * np.cos(az_rad)

        vec_sum = df_gnss.groupby("utcTimeMillis")[["_w_sin", "_w_cos"]].sum()
        ctx_features["AzimuthCentroid"] = np.degrees(
            np.arctan2(vec_sum["_w_sin"], vec_sum["_w_cos"])
        )

        # --- WLS Baseline ---
        # Take the first value per timestamp
        wls_cols = [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        wls_baseline = grouped_all[wls_cols].first()

        # --- Merge All Features ---
        # Start with metadata
        df_res = df_meta_drive.copy()
        df_res = df_res.set_index("UnixTimeMillis")

        # Join features
        dfs_to_join = [feat_global, feat_hp, feat_hr, ctx_features, wls_baseline]
        for d in dfs_to_join:
            if d is not None:
                df_res = df_res.join(d, how="left")

        # Reset index
        df_res = df_res.reset_index()

        # Fill NaNs for missing strata with 0
        feat_cols = [
            c
            for c in df_res.columns
            if any(x in c for x in Config.STRATA) or c in Config.GLOBAL_FEATURES
        ]
        df_res[feat_cols] = df_res[feat_cols].fillna(0)

        # --- Target Calculation ---
        wls_x = df_res["WlsPositionXEcefMeters"].values
        wls_y = df_res["WlsPositionYEcefMeters"].values
        wls_z = df_res["WlsPositionZEcefMeters"].values

        valid_wls = ~np.isnan(wls_x)

        wls_lat = np.zeros_like(wls_x)
        wls_lon = np.zeros_like(wls_x)
        wls_alt = np.zeros_like(wls_x)

        if np.any(valid_wls):
            lat, lon, alt = ecef_to_lla(
                wls_x[valid_wls], wls_y[valid_wls], wls_z[valid_wls]
            )
            wls_lat[valid_wls] = lat
            wls_lon[valid_wls] = lon
            wls_alt[valid_wls] = alt

        df_res["WlsLatitudeDegrees"] = wls_lat
        df_res["WlsLongitudeDegrees"] = wls_lon

        if is_train:
            # Calculate targets: ENU offset from WLS to GT
            gt_lat = df_res["LatitudeDegrees"].values
            gt_lon = df_res["LongitudeDegrees"].values

            # Use WLS altitude for GT to compute horizontal offset on the local plane
            t_east, t_north, _ = wgs84_to_enu(
                gt_lat, gt_lon, wls_alt, wls_lat, wls_lon, wls_alt
            )

            df_res["target_east"] = t_east
            df_res["target_north"] = t_north

            # Filter out rows where WLS was invalid
            df_res = df_res[valid_wls].copy()

        return df_res

    def process_data(self, split="train", load_cached_data=True):
        cache_path = self._get_cache_path(split)

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Processing {split} data...")

        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            is_train = True
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
            is_train = True
        else:
            meta_path = Config.TEST_METADATA_PATH
            is_train = False

        df_meta = pd.read_csv(meta_path)

        # Inject global split variable for _process_drive to resolve folder correctly if needed,
        # though _process_drive now checks both folders.
        global split_name
        split_name = split

        results = []
        groups = df_meta.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group_df in groups:
            processed_df = self._process_drive(drive_id, phone_name, group_df, is_train)
            if not processed_df.empty:
                results.append(processed_df)

        if not results:
            print("No data processed!")
            return pd.DataFrame()

        final_df = pd.concat(results, ignore_index=True)

        print(f"Saving {split} data to cache: {cache_path}")
        final_df.to_parquet(cache_path, index=False)

        return final_df
