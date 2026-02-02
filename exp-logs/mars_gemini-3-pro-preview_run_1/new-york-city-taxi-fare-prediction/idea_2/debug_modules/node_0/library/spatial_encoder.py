import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances_argmin
from library import config


class SpatialRouteEncoder:
    """
    Implements Spatially-Discretized Target Encoding.
    Cluters locations into neighborhoods and calculates historical average fares for routes.
    """

    def __init__(
        self,
        n_clusters=config.N_CLUSTERS,
        n_folds=config.K_FOLDS,
        cache_dir=config.CACHE_DIR,
        random_state=config.RANDOM_SEED,
    ):
        self.n_clusters = n_clusters
        self.n_folds = n_folds
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.cluster_centers_ = None

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_centers_path(self):
        return os.path.join(self.cache_dir, f"kmeans_centers_{self.n_clusters}.npy")

    def _get_route_map_path(self):
        return os.path.join(self.cache_dir, "route_fare_map.parquet")

    def _get_global_mean_path(self):
        return os.path.join(self.cache_dir, "global_mean_fare.txt")

    def fit_clusters(self, df, sample_size=500000, load_cached_data=True):
        """
        Fits MiniBatchKMeans on the coordinates of the dataframe to define spatial neighborhoods.
        Caches cluster centers to .npy file.
        """
        path = self._get_centers_path()

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(path):
            self.cluster_centers_ = np.load(path)
            return

        # 2. Compute from scratch
        # Stack pickup and dropoff to learn general city geography
        coords = np.vstack(
            (
                df[["pickup_latitude", "pickup_longitude"]].values,
                df[["dropoff_latitude", "dropoff_longitude"]].values,
            )
        )

        # Sample data if too large to speed up clustering
        if len(coords) > sample_size:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(coords), sample_size, replace=False)
            coords = coords[idx]

        # Fit KMeans
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=10000,
            n_init=3,
        )
        kmeans.fit(coords)
        self.cluster_centers_ = kmeans.cluster_centers_

        # 3. Save to cache
        np.save(path, self.cluster_centers_)

    def _assign_clusters(self, df):
        """
        Assigns pickup and dropoff cluster IDs to the dataframe using nearest centroid.
        """
        if self.cluster_centers_ is None:
            # Try loading
            path = self._get_centers_path()
            if os.path.exists(path):
                self.cluster_centers_ = np.load(path)
            else:
                raise ValueError(
                    "Cluster centers not initialized. Call fit_clusters first."
                )

        def predict_chunked(X, chunk_size=100000):
            labels = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i : i + chunk_size]
                # pairwise_distances_argmin is efficient and doesn't require the full KMeans object
                lab = pairwise_distances_argmin(chunk, self.cluster_centers_)
                labels.append(lab)
            return np.concatenate(labels)

        p_coords = df[["pickup_latitude", "pickup_longitude"]].values
        d_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

        return predict_chunked(p_coords), predict_chunked(d_coords)

    def get_oof_target_encoding(
        self, df, target_col="fare_amount", load_cached_data=True
    ):
        """
        Generates Out-of-Fold (OOF) target encoded features for the training set.
        Prevents leakage by calculating means on K-1 folds and applying to the Kth fold.
        """
        cache_path = os.path.join(self.cache_dir, "train_spatial_oof.parquet")

        if load_cached_data and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        # Assign clusters
        p_labels, d_labels = self._assign_clusters(df)

        # Create working dataframe
        tmp = pd.DataFrame(
            {"p": p_labels, "d": d_labels, "target": df[target_col].values}
        )

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(df), dtype=np.float32)
        global_mean = tmp["target"].mean()

        # Iterate folds
        for train_idx, val_idx in kf.split(tmp):
            train_fold = tmp.iloc[train_idx]
            val_fold = tmp.iloc[val_idx]

            # Calculate route means on training fold
            means = train_fold.groupby(["p", "d"])["target"].mean()

            # Map to validation fold
            # join with rsuffix to avoid collision if names overlap
            mapped = val_fold.join(means, on=["p", "d"], rsuffix="_mean")

            # 'target' is the name of the series from groupby, so it becomes 'target' column in join
            # If val_fold already had 'target', the joined one is 'target_mean' (if rsuffix used)
            # or we just access the Series logic.
            # To be safe with column names:
            if "target" in mapped.columns and isinstance(
                mapped["target"], pd.DataFrame
            ):
                # This happens if collision occurs and rsuffix wasn't applied to the index join properly
                # But join on index usually returns the series.
                # Let's be explicit with rename.
                pass

            # The series 'means' is named 'target'. val_fold has 'target'.
            # join(means, on=...) will likely raise error or require suffix.
            # Let's rename the series before join.
            means.name = "mean_fare"
            mapped = val_fold.join(means, on=["p", "d"])

            fold_preds = mapped["mean_fare"].fillna(global_mean).values
            oof_preds[val_idx] = fold_preds

        result = pd.DataFrame({"mean_fare_route": oof_preds}, index=df.index)

        # Save to cache
        result.to_parquet(cache_path)
        return result

    def fit_global_map(self, df, target_col="fare_amount", load_cached_data=True):
        """
        Computes the global route map (Route -> Mean Fare) from the full training set.
        Used for inference on Validation and Test sets.
        """
        map_path = self._get_route_map_path()
        mean_path = self._get_global_mean_path()

        if load_cached_data and os.path.exists(map_path) and os.path.exists(mean_path):
            return

        p_labels, d_labels = self._assign_clusters(df)
        tmp = pd.DataFrame(
            {"p": p_labels, "d": d_labels, "target": df[target_col].values}
        )

        # Compute Route Map
        route_map = tmp.groupby(["p", "d"])["target"].mean().reset_index()
        route_map.columns = ["p", "d", "mean_fare"]

        # Compute Global Mean
        global_mean = tmp["target"].mean()

        # Save
        route_map.to_parquet(map_path, index=False)
        with open(mean_path, "w") as f:
            f.write(str(global_mean))

    def get_global_target_encoding(self, df, split_name, load_cached_data=True):
        """
        Applies the global route map to a dataframe (Validation or Test).
        """
        cache_path = os.path.join(
            self.cache_dir, f"{split_name}_spatial_global.parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        map_path = self._get_route_map_path()
        mean_path = self._get_global_mean_path()

        if not os.path.exists(map_path):
            raise FileNotFoundError(
                "Global route map not found. Call fit_global_map with train set first."
            )

        # Load artifacts
        route_map = pd.read_parquet(map_path)
        with open(mean_path, "r") as f:
            global_mean = float(f.read().strip())

        # Assign clusters
        p_labels, d_labels = self._assign_clusters(df)
        tmp = pd.DataFrame({"p": p_labels, "d": d_labels})

        # Merge map
        merged = tmp.merge(route_map, on=["p", "d"], how="left")
        preds = merged["mean_fare"].fillna(global_mean).values

        result = pd.DataFrame({"mean_fare_route": preds}, index=df.index)

        # Save
        result.to_parquet(cache_path)
        return result
