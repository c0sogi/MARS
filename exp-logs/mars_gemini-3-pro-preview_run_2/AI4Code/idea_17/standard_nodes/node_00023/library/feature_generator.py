import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import joblib

from library.config import Config
from library.data_handler import NotebookLoader
from library.text_pipeline import TextVectorizers, FunctionalCodeClusterer


class FeatureFactory:
    """
    Orchestrates the generation of features for the Two-Stage Stacked Hybrid Ranking model.
    Implements Functional Landmark Triangulation and Neighborhood Smoothing.
    """

    def __init__(self):
        self.config = Config
        self.loader = NotebookLoader()
        self.text_pipeline = TextVectorizers()
        self.clusterer = FunctionalCodeClusterer()

        # Local SVD for Markdown LSA (Stage 2 feature)
        self.md_lsa_path = os.path.join(self.config.WORKING_DIR, "md_lsa_model.joblib")
        self.md_lsa = TruncatedSVD(
            n_components=32, random_state=self.config.RANDOM_STATE
        )

    def _get_data(self, split: str, load_cached_data: bool = True):
        """
        Loads dataframe and fixes code ranks for test set.
        """
        df_md, df_code = self.loader.load_data(
            split=split, load_cached_data=load_cached_data
        )

        # Fix Code Ranks for Test Set
        # The loader returns rank=-1 for test. We must infer relative order from dataframe order.
        if split == "test":
            # We assume df_code is sorted by notebook and then by occurrence (insertion order)
            # We calculate normalized rank: 0.0 to 1.0

            # Helper to calc rank
            def calc_rank(g):
                n = len(g)
                if n <= 1:
                    return pd.Series([0.0] * n, index=g.index)
                return pd.Series(np.arange(n) / (n - 1), index=g.index)

            # Apply group by
            new_ranks = df_code.groupby("notebook_id", group_keys=False)[
                "cell_id"
            ].apply(calc_rank)
            df_code["rank"] = new_ranks.astype(np.float32)

        return df_md, df_code

    def _fit_transform_text(
        self, split: str, df_md, df_code, load_cached_data: bool = True
    ):
        """
        Handles vectorization lifecycle. Fits on train, transforms on val/test.
        """
        md_text = df_md["source"].fillna("").values
        code_text = df_code["source"].fillna("").values

        if split == "train":
            self.text_pipeline.fit_markdown(md_text, load_cached_data=load_cached_data)
            self.text_pipeline.fit_code(code_text, load_cached_data=load_cached_data)

        # Transform
        md_tfidf = self.text_pipeline.transform_markdown(md_text)
        code_tfidf = self.text_pipeline.transform_code(code_text)

        return md_tfidf, code_tfidf

    def _get_code_clusters(self, split: str, code_tfidf, load_cached_data: bool = True):
        """
        Handles clustering lifecycle. Fits on train, predicts on val/test.
        """
        if split == "train":
            self.clusterer.fit(code_tfidf, load_cached_data=load_cached_data)

        return self.clusterer.predict_clusters(code_tfidf)

    def _extract_complex_features(
        self, df_md, df_code, md_tfidf, code_tfidf, code_clusters
    ):
        """
        Core engine: Functional Landmark Triangulation + Neighborhood Smoothing.
        Iterates over notebooks to compute features relative to the notebook's code skeleton.
        """
        # Prepare output arrays
        n_md = len(df_md)
        n_clusters = self.config.NUM_CODE_CLUSTERS

        # Features:
        # Landmarks: K clusters * 2 (Rank, Sim)
        # Neighborhood: 1 (Mean Rank)
        landmark_feats = np.zeros((n_md, n_clusters * 2), dtype=np.float32)
        neighborhood_feats = np.zeros((n_md, 1), dtype=np.float32)

        # Grouping for iteration
        # We need to map global indices to notebook groups
        md_groups = df_md.groupby("notebook_id")
        code_groups = df_code.groupby("notebook_id")

        # Create a mapping from notebook_id to code indices
        code_nb_to_indices = code_groups.indices
        md_nb_to_indices = md_groups.indices

        # Iterate over notebooks that exist in MD (some notebooks might have 0 MD cells?)
        # Intersection of keys
        common_ids = set(md_nb_to_indices.keys()) & set(code_nb_to_indices.keys())

        # We iterate silently (no tqdm)
        for nb_id in common_ids:
            md_idx = md_nb_to_indices[nb_id]
            code_idx = code_nb_to_indices[nb_id]

            if len(code_idx) == 0:
                continue

            # Slice TF-IDF matrices
            # md_sub: (n_md_nb, vocab), code_sub: (n_code_nb, vocab)
            md_sub = md_tfidf[md_idx]
            code_sub = code_tfidf[code_idx]

            # Compute Similarity Matrix (n_md_nb, n_code_nb)
            # Dense matrix is fine here as notebooks are usually small (<200 cells)
            sim_matrix = cosine_similarity(md_sub, code_sub)

            # Get Code Attributes
            local_code_ranks = df_code.iloc[code_idx]["rank"].values
            local_code_clusters = code_clusters[code_idx]

            # --- 1. Functional Landmark Triangulation ---
            for k in range(n_clusters):
                # Find columns (code cells) belonging to cluster k
                k_indices = np.where(local_code_clusters == k)[0]

                feat_idx_rank = k * 2
                feat_idx_sim = k * 2 + 1

                if len(k_indices) > 0:
                    # Extract sub-matrix for this cluster
                    sim_sub = sim_matrix[:, k_indices]

                    # Find best match for each MD cell
                    best_sims = np.max(sim_sub, axis=1)
                    best_args = np.argmax(sim_sub, axis=1)

                    # Map local argmax back to code rank
                    # k_indices[best_args] gives the index into local_code_ranks
                    best_ranks = local_code_ranks[k_indices[best_args]]

                    landmark_feats[md_idx, feat_idx_rank] = best_ranks
                    landmark_feats[md_idx, feat_idx_sim] = best_sims
                else:
                    # Cluster not present in this notebook
                    # Default: Rank -1 (missing), Sim 0
                    landmark_feats[md_idx, feat_idx_rank] = -1.0
                    landmark_feats[md_idx, feat_idx_sim] = 0.0

            # --- 2. Neighborhood Smoothing ---
            # Top-N similar code cells regardless of cluster
            N = self.config.NEIGHBORHOOD_SIZE
            if sim_matrix.shape[1] > 0:
                # If fewer than N code cells, take all
                k_top = min(N, sim_matrix.shape[1])

                # argpartition is faster than sort
                # We want indices of k largest elements
                # -sim_matrix because argpartition puts smallest first
                top_indices = np.argpartition(-sim_matrix, k_top - 1, axis=1)[:, :k_top]

                # Gather ranks
                # We need to broadcast or iterate.
                # Advanced indexing: local_code_ranks[top_indices] -> (n_md_nb, k_top)
                top_ranks = local_code_ranks[top_indices]

                # Compute mean
                neighborhood_feats[md_idx, 0] = np.mean(top_ranks, axis=1)
            else:
                neighborhood_feats[md_idx, 0] = 0.5  # Fallback

        return landmark_feats, neighborhood_feats

    def _get_markdown_lsa(self, split: str, md_tfidf, load_cached_data: bool = True):
        """
        Generates LSA features for markdown text.
        """
        if split == "train":
            if load_cached_data and os.path.exists(self.md_lsa_path):
                self.md_lsa = joblib.load(self.md_lsa_path)
                lsa_feats = self.md_lsa.transform(md_tfidf)
            else:
                lsa_feats = self.md_lsa.fit_transform(md_tfidf)
                joblib.dump(self.md_lsa, self.md_lsa_path)
        else:
            # For val/test, model must exist (fitted on train)
            if not os.path.exists(self.md_lsa_path):
                # Fallback if not trained (should not happen in proper pipeline)
                lsa_feats = self.md_lsa.fit_transform(md_tfidf)
            else:
                self.md_lsa = joblib.load(self.md_lsa_path)
                lsa_feats = self.md_lsa.transform(md_tfidf)

        return lsa_feats.astype(np.float32)

    def build_stage1_dataset(self, split: str, load_cached_data: bool = True):
        """
        Builds the sparse dataset for Stage 1 (Ridge Regression).

        Returns:
            X (sparse matrix): TF-IDF features.
            y (array): Normalized ranks.
            groups (array): Notebook IDs.
        """
        cache_path_X = os.path.join(self.config.WORKING_DIR, f"stage1_{split}_X.npz")
        cache_path_y = os.path.join(self.config.WORKING_DIR, f"stage1_{split}_y.npy")
        cache_path_g = os.path.join(
            self.config.WORKING_DIR, f"stage1_{split}_groups.npy"
        )

        if (
            load_cached_data
            and os.path.exists(cache_path_X)
            and os.path.exists(cache_path_y)
        ):
            print(f"Loading Stage 1 {split} data from cache...")
            X = sp.load_npz(cache_path_X)
            y = np.load(cache_path_y, allow_pickle=True)
            groups = np.load(cache_path_g, allow_pickle=True)
            return X, y, groups

        print(f"Building Stage 1 {split} data...")

        # 1. Load Data
        df_md, df_code = self._get_data(split, load_cached_data)

        # 2. Vectorize
        md_tfidf, _ = self._fit_transform_text(split, df_md, df_code, load_cached_data)

        # 3. Prepare Outputs
        X = md_tfidf
        y = df_md["rank"].values
        groups = df_md["notebook_id"].values

        # 4. Cache
        sp.save_npz(cache_path_X, X)
        np.save(cache_path_y, y)
        np.save(cache_path_g, groups)

        return X, y, groups

    def build_stage2_dataset(
        self, split: str, ridge_preds: np.ndarray, load_cached_data: bool = True
    ):
        """
        Builds the dense dataset for Stage 2 (LightGBM).

        Args:
            split: 'train', 'val', or 'test'.
            ridge_preds: Predictions from Stage 1 (OOF for train, direct for val/test).

        Returns:
            df_features (pd.DataFrame): Dense features including ridge_pred.
            y (array): Normalized ranks.
            groups (array): Notebook IDs.
        """
        cache_path = os.path.join(
            self.config.WORKING_DIR, f"stage2_{split}_features.parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading Stage 2 {split} data from cache...")
            df_features = pd.read_parquet(cache_path)
            # y and groups are stored in columns 'target' and 'notebook_id'
            y = df_features["target"].values
            groups = df_features["notebook_id"].values
            # Drop non-feature columns for return
            df_features = df_features.drop(columns=["target", "notebook_id"])
            return df_features, y, groups

        print(f"Building Stage 2 {split} data...")

        # 1. Load Data
        df_md, df_code = self._get_data(split, load_cached_data)

        # 2. Vectorize
        md_tfidf, code_tfidf = self._fit_transform_text(
            split, df_md, df_code, load_cached_data
        )

        # 3. Cluster Code Cells
        code_clusters = self._get_code_clusters(split, code_tfidf, load_cached_data)

        # 4. Extract Complex Features (Landmarks + Neighborhoods)
        print("Extracting Landmark & Neighborhood features...")
        landmark_feats, neighborhood_feats = self._extract_complex_features(
            df_md, df_code, md_tfidf, code_tfidf, code_clusters
        )

        # 5. Extract LSA Features
        print("Extracting LSA features...")
        lsa_feats = self._get_markdown_lsa(split, md_tfidf, load_cached_data)

        # 6. Construct DataFrame
        print("Assembling Stage 2 DataFrame...")

        # Base Metadata
        df_features = pd.DataFrame()
        df_features["notebook_id"] = df_md["notebook_id"]
        df_features["target"] = df_md["rank"]

        # Ridge Prediction (Stacked Feature)
        if len(ridge_preds) != len(df_md):
            raise ValueError(
                f"Length mismatch: Ridge preds {len(ridge_preds)} vs MD cells {len(df_md)}"
            )
        df_features["ridge_pred"] = ridge_preds

        # Simple Text Stats
        df_features["char_len"] = df_md["source"].str.len().fillna(0).astype(np.float32)
        df_features["word_len"] = (
            df_md["source"].apply(lambda x: len(str(x).split())).astype(np.float32)
        )

        # Notebook Context Stats
        # Map notebook_id to total code cells
        nb_code_counts = df_code.groupby("notebook_id").size()
        df_features["n_code_cells"] = (
            df_md["notebook_id"].map(nb_code_counts).fillna(0).astype(np.float32)
        )

        # Neighborhood
        df_features["neighborhood_rank"] = neighborhood_feats.flatten()

        # Landmarks
        for k in range(self.config.NUM_CODE_CLUSTERS):
            df_features[f"landmark_{k}_rank"] = landmark_feats[:, k * 2]
            df_features[f"landmark_{k}_sim"] = landmark_feats[:, k * 2 + 1]

        # LSA
        for i in range(lsa_feats.shape[1]):
            df_features[f"md_lsa_{i}"] = lsa_feats[:, i]

        # 7. Cache
        df_features.to_parquet(cache_path, index=False)

        # Return split
        y = df_features["target"].values
        groups = df_features["notebook_id"].values
        df_features = df_features.drop(columns=["target", "notebook_id"])

        return df_features, y, groups
