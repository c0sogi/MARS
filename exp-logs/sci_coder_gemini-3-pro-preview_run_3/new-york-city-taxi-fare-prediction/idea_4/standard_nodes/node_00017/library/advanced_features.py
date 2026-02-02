import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from library.config import Config


class SpatiotemporalEngine:
    """
    Implements Spatiotemporal Target Encoding using MiniBatchKMeans and K-Fold CV.
    Generates 'Average Rush Hour Price' features for specific neighborhoods.
    """

    def __init__(self):
        # Configuration
        self.n_clusters = Config.KMEANS_CLUSTERS
        self.n_folds = Config.TE_FOLDS
        self.seed = Config.SEED
        self.working_dir = Config.WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Artifact paths
        self.kmeans_path = Config.KMEANS_MODEL_PATH
        self.te_map_path = Config.TE_MAP_PATH

        # In-memory artifacts
        self.kmeans = None
        self.te_map = None

    def fit_transform_train(self, df, load_cached_data=True):
        """
        Fits KMeans on coordinates, performs 5-Fold CV Target Encoding on the training set,
        computes the global map for inference, and caches the processed dataframe.

        Args:
            df (pd.DataFrame): Training data containing coordinates, hour, and fare_amount.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The training dataframe with added cluster and target encoding columns.
        """
        # Define cache path for the processed training data
        cache_path = os.path.join(self.working_dir, "train_te.parquet")

        # 1. Try Loading Cache
        # We must ensure both the dataframe cache AND the model artifacts exist
        if load_cached_data and os.path.exists(cache_path):
            if os.path.exists(self.kmeans_path) and os.path.exists(self.te_map_path):
                print(f"Loading cached Spatiotemporal TE data from {cache_path}...")
                try:
                    cached_df = pd.read_parquet(cache_path)

                    # Verify cache integrity (simple length check against input)
                    # This prevents loading full cache when debugging with a sample
                    if len(cached_df) == len(df):
                        # Load artifacts into memory for consistency
                        self.kmeans = joblib.load(self.kmeans_path)
                        self.te_map = joblib.load(self.te_map_path)
                        return cached_df
                    else:
                        print(
                            "Cache size mismatch (likely due to sampling). Recomputing..."
                        )
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")
            else:
                print("Artifacts (KMeans/TE Map) missing. Recomputing...")

        print("Performing Spatiotemporal Target Encoding on Training Data...")

        # Work on a copy to avoid side effects on input
        df = df.copy()

        # 2. Fit KMeans (Spatial Clustering)
        print(f"Fitting MiniBatchKMeans with K={self.n_clusters}...")

        # Stack pickup and dropoff coordinates to learn a shared set of location clusters
        # This provides better coverage of the city geometry
        coords = np.vstack(
            (
                df[["pickup_latitude", "pickup_longitude"]].values,
                df[["dropoff_latitude", "dropoff_longitude"]].values,
            )
        )

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=10000,
            random_state=self.seed,
            n_init=3,
        )
        self.kmeans.fit(coords)

        # Save KMeans model
        joblib.dump(self.kmeans, self.kmeans_path)

        # 3. Assign Clusters
        print("Assigning spatial clusters...")
        # Predict clusters for pickup and dropoff locations
        df["pickup_cluster"] = self.kmeans.predict(
            df[["pickup_latitude", "pickup_longitude"]].values
        )
        df["dropoff_cluster"] = self.kmeans.predict(
            df[["dropoff_latitude", "dropoff_longitude"]].values
        )

        # 4. Create Composite Keys (Cluster + Hour)
        # We create a unique integer key for each (Cluster, Hour) pair
        # Formula: Cluster_ID * 24 + Hour (0-23)
        df["pickup_key"] = df["pickup_cluster"] * 24 + df["hour"]
        df["dropoff_key"] = df["dropoff_cluster"] * 24 + df["hour"]

        # 5. CV Target Encoding
        # We use K-Fold CV to generate target encodings for the training set to prevent leakage
        print(f"Performing {self.n_folds}-Fold CV Target Encoding...")
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # Initialize new features with NaN
        df["te_pickup"] = np.nan
        df["te_dropoff"] = np.nan

        # Reset index to ensure alignment during CV
        df = df.reset_index(drop=True)

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            # Extract data for this fold
            train_keys_p = df.loc[train_idx, "pickup_key"]
            train_keys_d = df.loc[train_idx, "dropoff_key"]
            train_y = df.loc[train_idx, "fare_amount"]

            # Compute means on the training part of the fold
            mean_map_p = train_y.groupby(train_keys_p).mean()
            mean_map_d = train_y.groupby(train_keys_d).mean()

            # Calculate global mean of the fold as fallback for unseen keys
            fold_global_mean = train_y.mean()

            # Map means to the validation part of the fold
            # fillna handles cases where a key in val_idx wasn't present in train_idx
            df.loc[val_idx, "te_pickup"] = (
                df.loc[val_idx, "pickup_key"].map(mean_map_p).fillna(fold_global_mean)
            )
            df.loc[val_idx, "te_dropoff"] = (
                df.loc[val_idx, "dropoff_key"].map(mean_map_d).fillna(fold_global_mean)
            )

        # 6. Compute Global Map for Inference
        # For Test/Validation sets, we use the means calculated from the FULL training set
        print("Computing Global Target Encoding Map...")
        global_map_p = df.groupby("pickup_key")["fare_amount"].mean().to_dict()
        global_map_d = df.groupby("dropoff_key")["fare_amount"].mean().to_dict()
        global_mean_all = df["fare_amount"].mean()

        self.te_map = {
            "pickup": global_map_p,
            "dropoff": global_map_d,
            "global_mean": global_mean_all,
        }

        # Save Global Map
        joblib.dump(self.te_map, self.te_map_path)

        # 7. Save Processed Data to Cache
        print("Saving processed training data to cache...")
        df.to_parquet(cache_path, index=False)

        return df

    def transform_test(self, df, load_cached_data=True):
        """
        Applies the learned Spatiotemporal Target Encoding to Test or Validation data.
        Uses the Global Map derived from the full training set.

        Args:
            df (pd.DataFrame): Data to transform.
            load_cached_data (bool): If True, ensures artifacts are loaded from disk.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        # 1. Load Artifacts
        # Ensure we have the fitted models in memory
        if self.kmeans is None or self.te_map is None:
            if (
                load_cached_data
                and os.path.exists(self.kmeans_path)
                and os.path.exists(self.te_map_path)
            ):
                self.kmeans = joblib.load(self.kmeans_path)
                self.te_map = joblib.load(self.te_map_path)
            else:
                raise FileNotFoundError(
                    "Artifacts not found. You must run fit_transform_train first."
                )

        print("Transforming Test/Validation Data...")
        df = df.copy()

        # 2. Assign Clusters
        # Use the pre-trained KMeans model
        df["pickup_cluster"] = self.kmeans.predict(
            df[["pickup_latitude", "pickup_longitude"]].values
        )
        df["dropoff_cluster"] = self.kmeans.predict(
            df[["dropoff_latitude", "dropoff_longitude"]].values
        )

        # 3. Create Composite Keys
        df["pickup_key"] = df["pickup_cluster"] * 24 + df["hour"]
        df["dropoff_key"] = df["dropoff_cluster"] * 24 + df["hour"]

        # 4. Map Values using Global Map
        global_mean = self.te_map["global_mean"]

        # Map and fill missing keys with the global average fare
        df["te_pickup"] = (
            df["pickup_key"].map(self.te_map["pickup"]).fillna(global_mean)
        )
        df["te_dropoff"] = (
            df["dropoff_key"].map(self.te_map["dropoff"]).fillna(global_mean)
        )

        return df
