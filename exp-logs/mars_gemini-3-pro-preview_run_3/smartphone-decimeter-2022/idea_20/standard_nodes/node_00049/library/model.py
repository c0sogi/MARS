import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import GroupKFold
from library.utils import CoordinateTransformer, MetricCalculator, IOHelper
from library.data_loader import GnssDataset
from library.features import FeatureEngineer


class ResidualRegressor:
    def __init__(
        self, n_estimators=2000, learning_rate=0.05, num_leaves=31, random_state=42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.models_east = []
        self.models_north = []
        self.feature_cols = []
        self.cache_dir = "./working/idea_20/"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_targets(self, df):
        """
        Compute ENU residuals (GT - WLS) for training data.
        """
        # 1. Get WLS ECEF
        wls_x = df["WlsPositionXEcefMeters"].values
        wls_y = df["WlsPositionYEcefMeters"].values
        wls_z = df["WlsPositionZEcefMeters"].values

        # 2. Get WLS Lat/Lon/Alt (Reference for ENU)
        ref_lat, ref_lon, ref_alt = CoordinateTransformer.ecef_to_wgs84(
            wls_x, wls_y, wls_z
        )

        # 3. Get GT ECEF
        # GT Altitude might be missing. Fill with WLS Altitude.
        gt_lat = df["LatitudeDegrees"].values
        gt_lon = df["LongitudeDegrees"].values
        gt_alt = df["AltitudeMeters"].fillna(pd.Series(ref_alt)).values

        gt_x, gt_y, gt_z = CoordinateTransformer.wgs84_to_ecef(gt_lat, gt_lon, gt_alt)

        # 4. Rotate difference to ENU
        e, n, u = CoordinateTransformer.ecef_to_enu(
            gt_x, gt_y, gt_z, ref_lat, ref_lon, ref_alt
        )

        df["target_east"] = e
        df["target_north"] = n
        df["target_up"] = u

        return df

    def prepare_data(self, mode="train", load_cached_data=True):
        """
        Load raw data, generate features, and compute targets (if train).
        """
        cache_file = f"prepared_{mode}_data.parquet"

        # Try load from cache
        if load_cached_data:
            df = IOHelper.load_parquet(cache_file)
            if df is not None:
                print(f"Loaded {mode} data from cache.")
                return df

        print(f"Preparing {mode} data from scratch...")

        # 1. Load Raw Data
        loader = GnssDataset(mode=mode)
        raw_df = loader.load(load_cached_data=load_cached_data)

        # 2. Feature Engineering
        fe = FeatureEngineer()

        processed_trips = []
        unique_trips = raw_df["tripId"].unique()

        for trip in unique_trips:
            trip_df = raw_df[raw_df["tripId"] == trip].copy()

            # FeatureEngineer calculates residuals and aggregates to epoch level.
            feat_df = fe.process_trip(trip_df, load_cached_data=load_cached_data)

            # Merge back with Ground Truth and other info present in trip_df
            # We take the first row per timestamp from raw data to get metadata/GT
            epoch_meta = trip_df.drop_duplicates(subset=["utcTimeMillis"])

            # Merge
            merged_trip = pd.merge(
                epoch_meta, feat_df, on=["tripId", "utcTimeMillis"], how="inner"
            )
            processed_trips.append(merged_trip)

        full_df = pd.concat(processed_trips, ignore_index=True)

        # 3. Compute Targets (only for train/val)
        if mode in ["train", "val"]:
            full_df = self._compute_targets(full_df)

        # Save to cache
        IOHelper.save_parquet(full_df, cache_file)

        return full_df

    def train_cross_validation(self, n_splits=5, load_cached_data=True):
        """
        Train models using GroupKFold cross-validation.
        """
        print("Starting Cross-Validation Training...")

        # Load Data
        df_train = self.prepare_data(mode="train", load_cached_data=load_cached_data)

        # Define Features
        exclude_cols = [
            "tripId",
            "utcTimeMillis",
            "drive_id",
            "phone_name",
            "LatitudeDegrees",
            "LongitudeDegrees",
            "AltitudeMeters",
            "SpeedMps",
            "AccuracyMeters",
            "BearingDegrees",
            "target_east",
            "target_north",
            "target_up",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "gnss_path",
            "imu_path",
            "gt_path",
            "MessageType",
            "MeasurementX",
            "MeasurementY",
            "MeasurementZ",
        ]

        self.feature_cols = [
            c
            for c in df_train.columns
            if c not in exclude_cols
            and df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")

        # Clean Features
        df_train[self.feature_cols] = df_train[self.feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        df_train[self.feature_cols] = df_train[self.feature_cols].fillna(0)

        X = df_train[self.feature_cols]
        y_east = df_train["target_east"]
        y_north = df_train["target_north"]
        groups = df_train["drive_id"]

        gkf = GroupKFold(n_splits=n_splits)

        scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_east, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            ye_train, ye_val = y_east.iloc[train_idx], y_east.iloc[val_idx]
            yn_train, yn_val = y_north.iloc[train_idx], y_north.iloc[val_idx]

            # Train East Model
            model_e = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                objective="mae",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model_e.fit(
                X_train,
                ye_train,
                eval_set=[(X_val, ye_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            self.models_east.append(model_e)

            # Train North Model
            model_n = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                objective="mae",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model_n.fit(
                X_train,
                yn_train,
                eval_set=[(X_val, yn_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            self.models_north.append(model_n)

            # Predict
            pred_e = model_e.predict(X_val)
            pred_n = model_n.predict(X_val)

            # Convert predictions back to Lat/Lon for scoring
            wls_x = df_train.iloc[val_idx]["WlsPositionXEcefMeters"].values
            wls_y = df_train.iloc[val_idx]["WlsPositionYEcefMeters"].values
            wls_z = df_train.iloc[val_idx]["WlsPositionZEcefMeters"].values

            ref_lat, ref_lon, ref_alt = CoordinateTransformer.ecef_to_wgs84(
                wls_x, wls_y, wls_z
            )

            # Assume Up error is 0
            pred_u = np.zeros_like(pred_e)

            pred_x, pred_y, pred_z = CoordinateTransformer.enu_to_ecef(
                pred_e, pred_n, pred_u, ref_lat, ref_lon, ref_alt
            )

            pred_lat, pred_lon, _ = CoordinateTransformer.ecef_to_wgs84(
                pred_x, pred_y, pred_z
            )

            # Prepare DF for scoring
            val_res = pd.DataFrame(
                {
                    "tripId": df_train.iloc[val_idx]["tripId"],
                    "UnixTimeMillis": df_train.iloc[val_idx]["utcTimeMillis"],
                    "LatitudeDegrees_pred": pred_lat,
                    "LongitudeDegrees_pred": pred_lon,
                    "LatitudeDegrees_gt": df_train.iloc[val_idx]["LatitudeDegrees"],
                    "LongitudeDegrees_gt": df_train.iloc[val_idx]["LongitudeDegrees"],
                }
            )

            score = MetricCalculator.calc_score(
                val_res.rename(
                    columns={
                        "LatitudeDegrees_pred": "LatitudeDegrees",
                        "LongitudeDegrees_pred": "LongitudeDegrees",
                    }
                ),
                val_res.rename(
                    columns={
                        "LatitudeDegrees_gt": "LatitudeDegrees",
                        "LongitudeDegrees_gt": "LongitudeDegrees",
                    }
                ),
            )
            scores.append(score)

            print(f"Fold {fold+1} Score: {score}")

        print(f"Mean CV Score: {np.mean(scores)}")

    def generate_submission(self, load_cached_data=True):
        """
        Generate predictions for test set and save submission file.
        """
        print("Generating Submission...")

        # Load Test Data
        df_test = self.prepare_data(mode="test", load_cached_data=load_cached_data)

        # Prepare Features
        df_test[self.feature_cols] = df_test[self.feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        df_test[self.feature_cols] = df_test[self.feature_cols].fillna(0)
        X_test = df_test[self.feature_cols]

        # Predict (Average of all fold models)
        pred_e = np.zeros(len(df_test))
        pred_n = np.zeros(len(df_test))

        for model in self.models_east:
            pred_e += model.predict(X_test)
        pred_e /= len(self.models_east)

        for model in self.models_north:
            pred_n += model.predict(X_test)
        pred_n /= len(self.models_north)

        # Convert to Lat/Lon
        wls_x = df_test["WlsPositionXEcefMeters"].values
        wls_y = df_test["WlsPositionYEcefMeters"].values
        wls_z = df_test["WlsPositionZEcefMeters"].values

        ref_lat, ref_lon, ref_alt = CoordinateTransformer.ecef_to_wgs84(
            wls_x, wls_y, wls_z
        )

        pred_u = np.zeros_like(pred_e)

        pred_x, pred_y, pred_z = CoordinateTransformer.enu_to_ecef(
            pred_e, pred_n, pred_u, ref_lat, ref_lon, ref_alt
        )

        pred_lat, pred_lon, _ = CoordinateTransformer.ecef_to_wgs84(
            pred_x, pred_y, pred_z
        )

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "tripId": df_test["tripId"],
                "UnixTimeMillis": df_test["utcTimeMillis"],
                "LatitudeDegrees": pred_lat,
                "LongitudeDegrees": pred_lon,
            }
        )

        # Ensure directory exists
        os.makedirs("./submission", exist_ok=True)
        submission.to_csv("./submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")
