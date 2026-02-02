import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import setup_logger, load_numpy, save_numpy, seed_everything
from library.feature_extractor import DeepFeatureExtractor


class LeafDataManager:
    """
    Manages data loading, stratification, and Manifold Densification.
    """

    def __init__(self):
        self.logger = setup_logger("DataManager")
        self.extractor = DeepFeatureExtractor()

    def _get_tabular_features(self, df):
        """
        Extracts the 192 tabular features (margin, shape, texture) from the dataframe.
        """
        # Identify columns
        cols = [
            c
            for c in df.columns
            if c.startswith("margin")
            or c.startswith("shape")
            or c.startswith("texture")
        ]
        # Ensure correct order and type
        return df[cols].values.astype(np.float32)

    def _densify(self, ids, dino, conv, tab, labels=None):
        """
        Transforms (N, 12, D) -> (3N, D) by averaging orthogonal views.
        Replicates tabular features, IDs, and labels.

        Args:
            ids: (N,)
            dino: (N, 12, 1024)
            conv: (N, 12, 1536)
            tab: (N, 192)
            labels: (N,) or None

        Returns:
            Tuple of densified arrays (ids, dino, conv, tab, labels)
        """
        n_samples = len(ids)
        n_centroids = Config.N_CENTROIDS  # 3

        dino_list = []
        conv_list = []

        # Generate 3 orthogonal centroids
        # Centroid k uses views: k, k+3, k+6, k+9
        for k in range(n_centroids):
            # Slice views for this centroid
            d_views = dino[:, k::n_centroids, :]  # (N, 4, D)
            c_views = conv[:, k::n_centroids, :]  # (N, 4, D)

            # Average over the views (axis 1)
            d_mean = np.mean(d_views, axis=1)  # (N, D)
            c_mean = np.mean(c_views, axis=1)  # (N, D)

            dino_list.append(d_mean)
            conv_list.append(c_mean)

        # Concatenate centroids: [Centroid_A_All; Centroid_B_All; Centroid_C_All]
        densified_dino = np.concatenate(dino_list, axis=0)  # (3N, D)
        densified_conv = np.concatenate(conv_list, axis=0)  # (3N, D)

        # Replicate invariant data
        densified_ids = np.tile(ids, n_centroids)  # (3N,)
        densified_tab = np.tile(tab, (n_centroids, 1))  # (3N, 192)

        densified_labels = None
        if labels is not None:
            densified_labels = np.tile(labels, n_centroids)  # (3N,)

        return (
            densified_ids,
            densified_dino,
            densified_conv,
            densified_tab,
            densified_labels,
        )

    def get_fold_data(self, fold_idx, load_cached_data=True):
        """
        Retrieves the densified training and validation data for a specific fold.

        Returns:
            (train_data_dict, train_labels), (val_data_dict, val_labels)
            Dictionaries contain keys: 'dino', 'conv', 'tab', 'ids'
        """
        prefix = f"fold_{fold_idx}"

        # Helper to load a set from cache
        def load_set(p):
            dino = load_numpy(f"{p}_dino.npy")
            conv = load_numpy(f"{p}_conv.npy")
            tab = load_numpy(f"{p}_tab.npy")
            ids = load_numpy(f"{p}_ids.npy")
            lbl = load_numpy(f"{p}_labels.npy")

            if any(x is None for x in [dino, conv, tab, ids, lbl]):
                return None
            return {"dino": dino, "conv": conv, "tab": tab, "ids": ids}, lbl

        # 1. Try loading from cache
        if load_cached_data:
            train_res = load_set(f"{prefix}_train")
            val_res = load_set(f"{prefix}_val")
            if train_res and val_res:
                self.logger.info(f"Loaded fold {fold_idx} data from cache.")
                return train_res[0], train_res[1], val_res[0], val_res[1]

        # 2. Generate from scratch
        self.logger.info(f"Generating densified data for fold {fold_idx}...")

        # Load all labeled data (Train + Val metadata) to maximize CV usage
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_all = pd.concat([df_train, df_val], ignore_index=True)

        # Extract features for all labeled data
        # Note: 'combined_train_val' is the cache key for the raw features
        ids, dino, conv = self.extractor.extract_features(
            df_all, "combined_train_val", load_cached_data=True
        )
        tab_features = self._get_tabular_features(df_all)
        labels = df_all["species"].values

        # Stratified Split
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Find the indices for the requested fold
        train_idx, val_idx = None, None
        for i, (t_idx, v_idx) in enumerate(skf.split(ids, labels)):
            if i == fold_idx:
                train_idx = t_idx
                val_idx = v_idx
                break

        if train_idx is None:
            raise ValueError(f"Fold {fold_idx} is out of range.")

        # Slice raw data
        tr_ids, val_ids = ids[train_idx], ids[val_idx]
        tr_dino, val_dino = dino[train_idx], dino[val_idx]
        tr_conv, val_conv = conv[train_idx], conv[val_idx]
        tr_tab, val_tab = tab_features[train_idx], tab_features[val_idx]
        tr_lbl, val_lbl = labels[train_idx], labels[val_idx]

        # Apply Manifold Densification
        # Both Train and Val are densified (3 centroids per image)
        d_tr_ids, d_tr_dino, d_tr_conv, d_tr_tab, d_tr_lbl = self._densify(
            tr_ids, tr_dino, tr_conv, tr_tab, tr_lbl
        )
        d_val_ids, d_val_dino, d_val_conv, d_val_tab, d_val_lbl = self._densify(
            val_ids, val_dino, val_conv, val_tab, val_lbl
        )

        # Save to cache
        def save_set(p, d_ids, d_dino, d_conv, d_tab, d_lbl):
            save_numpy(d_ids, f"{p}_ids.npy")
            save_numpy(d_dino, f"{p}_dino.npy")
            save_numpy(d_conv, f"{p}_conv.npy")
            save_numpy(d_tab, f"{p}_tab.npy")
            save_numpy(d_lbl, f"{p}_labels.npy")

        save_set(f"{prefix}_train", d_tr_ids, d_tr_dino, d_tr_conv, d_tr_tab, d_tr_lbl)
        save_set(
            f"{prefix}_val", d_val_ids, d_val_dino, d_val_conv, d_val_tab, d_val_lbl
        )

        return (
            {"dino": d_tr_dino, "conv": d_tr_conv, "tab": d_tr_tab, "ids": d_tr_ids},
            d_tr_lbl,
            {
                "dino": d_val_dino,
                "conv": d_val_conv,
                "tab": d_val_tab,
                "ids": d_val_ids,
            },
            d_val_lbl,
        )

    def get_test_data(self, load_cached_data=True):
        """
        Retrieves the densified test data (3 centroids per image).
        """
        prefix = "test_densified"

        # 1. Try loading from cache
        if load_cached_data:
            dino = load_numpy(f"{prefix}_dino.npy")
            conv = load_numpy(f"{prefix}_conv.npy")
            tab = load_numpy(f"{prefix}_tab.npy")
            ids = load_numpy(f"{prefix}_ids.npy")

            if all(x is not None for x in [dino, conv, tab, ids]):
                self.logger.info("Loaded test data from cache.")
                return {"dino": dino, "conv": conv, "tab": tab, "ids": ids}, ids

        # 2. Generate from scratch
        self.logger.info("Generating densified test data...")
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # Extract raw features
        ids, dino, conv = self.extractor.extract_features(
            df_test, "test", load_cached_data=True
        )
        tab_features = self._get_tabular_features(df_test)

        # Densify (No labels for test)
        d_ids, d_dino, d_conv, d_tab, _ = self._densify(
            ids, dino, conv, tab_features, labels=None
        )

        # Save to cache
        save_numpy(d_ids, f"{prefix}_ids.npy")
        save_numpy(d_dino, f"{prefix}_dino.npy")
        save_numpy(d_conv, f"{prefix}_conv.npy")
        save_numpy(d_tab, f"{prefix}_tab.npy")

        return {"dino": d_dino, "conv": d_conv, "tab": d_tab, "ids": d_ids}, d_ids
