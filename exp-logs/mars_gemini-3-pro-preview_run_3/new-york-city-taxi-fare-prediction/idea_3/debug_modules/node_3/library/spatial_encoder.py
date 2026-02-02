import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from library.config import N_CLUSTERS, SEED, CACHE_DIR, MIN_FARE, MAX_FARE


class SpatialTargetEncoder:
    """
    Encodes spatial coordinates (latitude/longitude) into mean target values
    using K-Means clustering and Target Encoding.
    """

    def __init__(self, n_clusters=N_CLUSTERS, random_state=SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state, batch_size=10000, n_init=3
        )
        self.pickup_means = {}
        self.dropoff_means = {}
        self.global_mean = 0.0

    def _get_coords(self, df):
        """Extracts pickup and dropoff coordinates."""
        pickup = df[["pickup_latitude", "pickup_longitude"]].fillna(0).values
        dropoff = df[["dropoff_latitude", "dropoff_longitude"]].fillna(0).values
        return pickup, dropoff

    def fit(self, df, y):
        """
        Fits the KMeans clusterer and computes global mean mappings.
        Used for preparing the encoder for inference (transform).
        """
        pickup_coords, dropoff_coords = self._get_coords(df)

        # Fit KMeans on all coordinates to define shared regions
        all_coords = np.vstack([pickup_coords, dropoff_coords])
        self.kmeans.fit(all_coords)

        # Predict clusters
        p_labels = self.kmeans.predict(pickup_coords)
        d_labels = self.kmeans.predict(dropoff_coords)

        # Compute Global Mean
        self.global_mean = np.mean(y)

        # Compute Cluster Means
        # Create temporary dataframe for aggregation
        temp_df = pd.DataFrame(
            {"p_cluster": p_labels, "d_cluster": d_labels, "target": y}
        )

        self.pickup_means = temp_df.groupby("p_cluster")["target"].mean().to_dict()
        self.dropoff_means = temp_df.groupby("d_cluster")["target"].mean().to_dict()

        return self

    def transform(self, df):
        """
        Applies the learned encodings to a dataframe.
        """
        pickup_coords, dropoff_coords = self._get_coords(df)

        # Predict clusters
        p_labels = self.kmeans.predict(pickup_coords)
        d_labels = self.kmeans.predict(dropoff_coords)

        # Map means
        # Use vectorized map or apply. Pandas map is efficient for series.
        p_encoded = (
            pd.Series(p_labels).map(self.pickup_means).fillna(self.global_mean).values
        )
        d_encoded = (
            pd.Series(d_labels).map(self.dropoff_means).fillna(self.global_mean).values
        )

        # Return dataframe with new features
        result = df.copy()
        result["pickup_cluster_fare"] = p_encoded
        result["dropoff_cluster_fare"] = d_encoded

        return result

    def fit_transform(self, df, y, n_splits=5):
        """
        Fits the encoder and transforms the training data using K-Fold OOF
        to prevent data leakage.
        """
        # 1. Fit KMeans on the geometry of the data
        pickup_coords, dropoff_coords = self._get_coords(df)
        all_coords = np.vstack([pickup_coords, dropoff_coords])
        self.kmeans.fit(all_coords)

        # 2. Prepare output arrays
        p_encoded_oof = np.zeros(len(df))
        d_encoded_oof = np.zeros(len(df))

        # 3. K-Fold Target Encoding
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # We need reset index for proper alignment if df index is not RangeIndex
        df_reset = df.reset_index(drop=True)
        y_reset = pd.Series(y).reset_index(drop=True)

        # Pre-calculate clusters for the whole dataset to speed up loop
        # (We already fit kmeans, so we can predict)
        all_p_labels = self.kmeans.predict(pickup_coords)
        all_d_labels = self.kmeans.predict(dropoff_coords)

        for train_idx, val_idx in kf.split(df_reset):
            # Get train/val subsets for this fold
            y_train = y_reset.iloc[train_idx]

            p_labels_train = all_p_labels[train_idx]
            d_labels_train = all_d_labels[train_idx]

            p_labels_val = all_p_labels[val_idx]
            d_labels_val = all_d_labels[val_idx]

            # Compute means on fold training set
            fold_global_mean = y_train.mean()

            # Groupby using pandas
            p_means = (
                pd.DataFrame({"c": p_labels_train, "y": y_train})
                .groupby("c")["y"]
                .mean()
            )
            d_means = (
                pd.DataFrame({"c": d_labels_train, "y": y_train})
                .groupby("c")["y"]
                .mean()
            )

            # Map to validation set
            p_encoded_oof[val_idx] = (
                pd.Series(p_labels_val).map(p_means).fillna(fold_global_mean).values
            )
            d_encoded_oof[val_idx] = (
                pd.Series(d_labels_val).map(d_means).fillna(fold_global_mean).values
            )

        # 4. Store global means for future inference (transform)
        self.fit(df, y)

        # 5. Attach features
        result = df.copy()
        result["pickup_cluster_fare"] = p_encoded_oof
        result["dropoff_cluster_fare"] = d_encoded_oof

        return result


def process_spatial_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Orchestrates the spatial feature engineering process with caching.
    """
    # Define cache paths
    train_out_path = os.path.join(CACHE_DIR, "train_spatial.parquet")
    val_out_path = os.path.join(CACHE_DIR, "val_spatial.parquet")
    test_out_path = os.path.join(CACHE_DIR, "test_spatial.parquet")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_out_path)
            and os.path.exists(val_out_path)
            and os.path.exists(test_out_path)
        ):
            print("Loading cached spatial features...")
            try:
                t = pd.read_parquet(train_out_path)
                v = pd.read_parquet(val_out_path)
                s = pd.read_parquet(test_out_path)
                return t, v, s
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

    print("Computing spatial features...")

    # Sanitize Training Data for Encoding
    # We filter outliers to ensure the mean calculations are robust
    mask = (train_df["fare_amount"] >= MIN_FARE) & (train_df["fare_amount"] <= MAX_FARE)
    clean_train_df = train_df[mask].copy()
    clean_y = clean_train_df["fare_amount"].values

    # Initialize Encoder
    encoder = SpatialTargetEncoder(n_clusters=N_CLUSTERS, random_state=SEED)

    # Fit and Transform Train (OOF)
    # Note: We fit on the cleaned data, but we might want to return the full train_df with features.
    # However, for consistency, we usually train on the cleaned dataset anyway.
    # Let's return the cleaned dataset with features to be safe and consistent with the idea.
    print("Encoding training set (OOF)...")
    train_processed = encoder.fit_transform(clean_train_df, clean_y)

    # Transform Val and Test
    # The encoder's internal state is now set to the full clean_train_df means (via the self.fit call at end of fit_transform)
    print("Encoding validation and test sets...")
    val_processed = encoder.transform(val_df)
    test_processed = encoder.transform(test_df)

    # Save to cache
    print("Saving to cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_processed.to_parquet(train_out_path, index=False)
    val_processed.to_parquet(val_out_path, index=False)
    test_processed.to_parquet(test_out_path, index=False)

    return train_processed, val_processed, test_processed
