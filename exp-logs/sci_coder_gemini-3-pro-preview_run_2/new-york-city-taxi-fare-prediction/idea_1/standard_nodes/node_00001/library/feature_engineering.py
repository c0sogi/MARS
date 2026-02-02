import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


class FeatureProcessor:
    """
    Handles data cleaning, feature engineering, and normalization for NYC Taxi Fare Prediction.
    Implements caching to speed up iterative development.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = [
            "dist_haversine",
            "dist_manhattan",
            "pickup_year",
            "pickup_hour",
            "pickup_JFK",
            "dropoff_JFK",
            "pickup_LGA",
            "dropoff_LGA",
            "pickup_EWR",
            "dropoff_EWR",
        ]

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates the great circle distance between two points in km."""
        R = 6371  # Earth radius in km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def _manhattan_distance(self, lat1, lon1, lat2, lon2):
        """Calculates the Manhattan distance (L1 norm) approximation in km."""
        # Approximation: 1 deg lat ~ 111 km. 1 deg lon ~ 85 km at 40 deg lat.
        return 111 * np.abs(lat1 - lat2) + 85 * np.abs(lon1 - lon2)

    def _flag_airports(self, lat, lon):
        """Generates binary flags for proximity to major NYC airports."""
        flags = {}
        for code, coords in Config.AIRPORTS.items():
            # Euclidean distance in degrees is sufficient for small radius checks
            d = np.sqrt((lat - coords["lat"]) ** 2 + (lon - coords["lon"]) ** 2)
            flags[code] = (d < Config.AIRPORT_RADIUS).astype(np.float32)
        return flags

    def _engineer_features(self, df):
        """Core feature engineering logic applied to both train and test sets."""
        # Extract coordinates
        plat = df["pickup_latitude"].values
        plon = df["pickup_longitude"].values
        dlat = df["dropoff_latitude"].values
        dlon = df["dropoff_longitude"].values

        # 1. Distance Features
        dist_hav = self._haversine_distance(plat, plon, dlat, dlon)
        dist_man = self._manhattan_distance(plat, plon, dlat, dlon)

        # 2. Time Features
        # Efficiently convert to datetime if not already
        if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
            dt_series = pd.to_datetime(df["pickup_datetime"], utc=True)
        else:
            dt_series = df["pickup_datetime"]

        year = dt_series.dt.year.values.astype(np.float32)
        hour = dt_series.dt.hour.values.astype(np.float32)

        # 3. Airport Flags
        p_flags = self._flag_airports(plat, plon)
        d_flags = self._flag_airports(dlat, dlon)

        # Stack features into a matrix
        # Order: hav, man, year, hour, p_JFK, d_JFK, p_LGA, d_LGA, p_EWR, d_EWR
        features = np.column_stack(
            [
                dist_hav,
                dist_man,
                year,
                hour,
                p_flags["JFK"],
                d_flags["JFK"],
                p_flags["LGA"],
                d_flags["LGA"],
                p_flags["EWR"],
                d_flags["EWR"],
            ]
        )

        return features.astype(np.float32)

    def _save_scaler(self):
        """Saves scaler parameters to cache to avoid pickling."""
        np.save(os.path.join(Config.CACHE_DIR, "scaler_mean.npy"), self.scaler.mean_)
        np.save(os.path.join(Config.CACHE_DIR, "scaler_scale.npy"), self.scaler.scale_)
        np.save(os.path.join(Config.CACHE_DIR, "scaler_var.npy"), self.scaler.var_)

    def _load_scaler(self):
        """Loads scaler parameters from cache."""
        try:
            mean = np.load(os.path.join(Config.CACHE_DIR, "scaler_mean.npy"))
            scale = np.load(os.path.join(Config.CACHE_DIR, "scaler_scale.npy"))
            var = np.load(os.path.join(Config.CACHE_DIR, "scaler_var.npy"))
            self.scaler.mean_ = mean
            self.scaler.scale_ = scale
            self.scaler.var_ = var
            self.scaler.n_samples_seen_ = 0  # Not strictly needed for transform
            return True
        except FileNotFoundError:
            return False

    def fit_transform(self, df, load_cached_data=True):
        """
        Cleans data, engineers features, fits scaler, and transforms training data.
        Returns X (features) and y (target).
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_X = os.path.join(Config.CACHE_DIR, "X_train.npy")
        cache_y = os.path.join(Config.CACHE_DIR, "y_train.npy")

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_y):
            if self._load_scaler():
                print("Loading cached training data...")
                X = np.load(cache_X)
                y = np.load(cache_y)
                return X, y

        print("Processing training data from scratch...")

        # 1. Data Cleaning (Filtering)
        # Apply constraints from Config
        mask = (
            (df["pickup_longitude"].between(Config.MIN_LON, Config.MAX_LON))
            & (df["pickup_latitude"].between(Config.MIN_LAT, Config.MAX_LAT))
            & (df["dropoff_longitude"].between(Config.MIN_LON, Config.MAX_LON))
            & (df["dropoff_latitude"].between(Config.MIN_LAT, Config.MAX_LAT))
            & (df["fare_amount"].between(Config.MIN_FARE, Config.MAX_FARE))
            & (
                df["passenger_count"].between(
                    Config.MIN_PASSENGERS, Config.MAX_PASSENGERS
                )
            )
        )
        df_clean = df[mask].copy()

        # 2. Feature Engineering
        X_raw = self._engineer_features(df_clean)
        y = df_clean["fare_amount"].values.astype(np.float32)

        # 3. Scaling
        X = self.scaler.fit_transform(X_raw)

        # 4. Save to Cache
        np.save(cache_X, X)
        np.save(cache_y, y)
        self._save_scaler()

        return X, y

    def transform(self, df, load_cached_data=True, cache_name="test"):
        """
        Engineers features and applies existing scaler to validation/test data.
        Does NOT filter rows (preserves test set integrity).
        Returns X (features).
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_X = os.path.join(Config.CACHE_DIR, f"X_{cache_name}.npy")

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_X):
            # Ensure scaler is loaded so subsequent calls work if needed
            self._load_scaler()
            print(f"Loading cached {cache_name} data...")
            return np.load(cache_X)

        print(f"Processing {cache_name} data from scratch...")

        # Ensure scaler is fitted
        if not hasattr(self.scaler, "mean_") or self.scaler.mean_ is None:
            if not self._load_scaler():
                raise RuntimeError(
                    "Scaler not fitted and no cache found. Run fit_transform first."
                )

        # Feature Engineering
        X_raw = self._engineer_features(df)

        # Scaling
        X = self.scaler.transform(X_raw)

        # Save to Cache
        np.save(cache_X, X)

        return X
