import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class GlobalVectorizer:
    """
    Manages the global TF-IDF and SVD models for text representation.
    Handles fitting, transforming, and persistence/caching of models.
    """

    def __init__(self):
        self.tfidf = None
        self.svd = None
        self.tfidf_path = Config.PATH_TFIDF_VECTORIZER
        self.svd_path = Config.PATH_SVD_MODEL

    def fit(self, text_series):
        """
        Fits the TF-IDF and SVD models on the provided text corpus.
        If models already exist on disk, they are loaded instead of re-fitting.
        """
        # Check if models exist
        if os.path.exists(self.tfidf_path) and os.path.exists(self.svd_path):
            print("Loading pre-trained Vectorizer and SVD models...")
            self.tfidf = joblib.load(self.tfidf_path)
            self.svd = joblib.load(self.svd_path)
            return

        print("Fitting Global TF-IDF Vectorizer...")
        self.tfidf = TfidfVectorizer(**Config.TFIDF_PARAMS)
        tfidf_matrix = self.tfidf.fit_transform(text_series.astype(str))

        print(f"Fitting Truncated SVD (Components={Config.SVD_COMPONENTS})...")
        self.svd = TruncatedSVD(
            n_components=Config.SVD_COMPONENTS, random_state=Config.SVD_RANDOM_STATE
        )
        self.svd.fit(tfidf_matrix)

        # Save models
        print("Saving models to disk...")
        joblib.dump(self.tfidf, self.tfidf_path)
        joblib.dump(self.svd, self.svd_path)

    def transform(self, text_series):
        """
        Transforms text into Sparse TF-IDF and Dense SVD representations.

        Returns:
            tuple: (sparse_matrix, dense_matrix)
        """
        if self.tfidf is None or self.svd is None:
            # Try loading if not in memory
            if os.path.exists(self.tfidf_path) and os.path.exists(self.svd_path):
                self.tfidf = joblib.load(self.tfidf_path)
                self.svd = joblib.load(self.svd_path)
            else:
                raise ValueError("Models not fitted or found. Call fit() first.")

        # Transform
        sparse_matrix = self.tfidf.transform(text_series.astype(str))
        dense_matrix = self.svd.transform(sparse_matrix)

        return sparse_matrix, dense_matrix


class MultiViewExtractor:
    """
    Implements the Multi-View Instance-Based Feature Engineering.
    Extracts Lexical, Latent, and Symbolic features describing the relationship
    between markdown cells and the code cell skeleton.
    """

    def __init__(self):
        self.vectorizer = GlobalVectorizer()
        # Regex for symbolic extraction (variables, functions)
        self.token_pattern = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    def _extract_tokens(self, text):
        """Extracts a set of unique identifiers from text."""
        if not isinstance(text, str):
            return set()
        return set(self.token_pattern.findall(text))

    def _compute_jaccard(self, set_a, set_b):
        """Computes Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def _get_notebook_features(self, group, sparse_matrix, dense_matrix):
        """
        Computes features for a single notebook group.
        """
        # Separate Code and Markdown
        code_mask = group["cell_type"] == "code"
        md_mask = group["cell_type"] == "markdown"

        code_indices = np.where(code_mask)[0]
        md_indices = np.where(md_mask)[0]

        # If no code cells, return empty or default features for MD cells
        # (Though competition notebooks usually have code)
        if len(code_indices) == 0:
            # Return rows with NaNs for features
            return []

        # Get vectors for this notebook
        # Note: sparse_matrix/dense_matrix are aligned with 'group' index reset
        nb_sparse = sparse_matrix
        nb_dense = dense_matrix

        # Pre-compute symbolic sets for all cells in this notebook
        sources = group["source"].tolist()
        symbolic_sets = [self._extract_tokens(s) for s in sources]

        features_list = []

        # Total code cells for normalization
        n_code = len(code_indices)

        # Pre-calculate Code Representations
        code_sparse = nb_sparse[code_indices]
        code_dense = nb_dense[code_indices]
        code_sets = [symbolic_sets[i] for i in code_indices]

        # Iterate over Markdown cells to generate features
        for rel_idx in md_indices:
            md_row = group.iloc[rel_idx]
            cell_id = md_row["cell_id"]

            # --- View 1: Lexical (Sparse Cosine) ---
            md_vec_sparse = nb_sparse[rel_idx]
            # Compute cosine similarity (1 x N_code)
            # cosine_similarity accepts sparse matrices
            lex_sims = cosine_similarity(md_vec_sparse, code_sparse)[0]

            # --- View 2: Latent (Dense Cosine) ---
            md_vec_dense = nb_dense[rel_idx].reshape(1, -1)
            lat_sims = cosine_similarity(md_vec_dense, code_dense)[0]

            # --- View 3: Symbolic (Jaccard) ---
            md_set = symbolic_sets[rel_idx]
            sym_sims = np.array(
                [self._compute_jaccard(md_set, c_set) for c_set in code_sets]
            )

            # --- Feature Extraction per View ---
            row_feats = {"id": md_row["id"], "cell_id": cell_id}

            views = {"lex": lex_sims, "lat": lat_sims, "sym": sym_sims}

            for view_name, sims in views.items():
                # Sort indices by similarity descending
                sorted_args = np.argsort(sims)[::-1]
                sorted_sims = sims[sorted_args]

                # Explicit Neighbors (Top K_EXPLICIT)
                for k in range(Config.K_NEIGHBORS_EXPLICIT):
                    if k < n_code:
                        # Rank is the normalized position of the code cell in the notebook
                        # code_indices[sorted_args[k]] is the relative index in the group,
                        # but we want the rank among code cells: sorted_args[k] is the index in code_indices array
                        # which corresponds to the 0..N-1 position of that code cell.
                        rank_norm = sorted_args[k] / (n_code - 1) if n_code > 1 else 0.0
                        score = sorted_sims[k]
                    else:
                        rank_norm = 0.5  # Default neutral rank
                        score = 0.0

                    row_feats[f"{view_name}_n{k+1}_rank"] = rank_norm
                    row_feats[f"{view_name}_n{k+1}_score"] = score

                # Smoothed Aggregates (Top K_SMOOTH)
                k_smooth = min(Config.K_NEIGHBORS_SMOOTH, n_code)
                if k_smooth > 0:
                    top_k_sims = sorted_sims[:k_smooth]
                    # Get ranks of top k
                    top_k_ranks = (
                        sorted_args[:k_smooth] / (n_code - 1)
                        if n_code > 1
                        else np.zeros(k_smooth)
                    )

                    row_feats[
                        f"{view_name}_top{Config.K_NEIGHBORS_SMOOTH}_mean_score"
                    ] = np.mean(top_k_sims)
                    row_feats[
                        f"{view_name}_top{Config.K_NEIGHBORS_SMOOTH}_std_score"
                    ] = np.std(top_k_sims)
                    row_feats[
                        f"{view_name}_top{Config.K_NEIGHBORS_SMOOTH}_mean_rank"
                    ] = np.mean(top_k_ranks)
                else:
                    row_feats[
                        f"{view_name}_top{Config.K_NEIGHBORS_SMOOTH}_mean_score"
                    ] = 0.0
                    row_feats[
                        f"{view_name}_top{Config.K_NEIGHBORS_SMOOTH}_std_score"
                    ] = 0.0
                    row_feats[
                        f"{view_name}_top{Config.K_NEIGHBORS_SMOOTH}_mean_rank"
                    ] = 0.5

            features_list.append(row_feats)

        return features_list

    def generate_features(self, df, split, load_cached_data=True):
        """
        Main method to generate features for a dataset split.
        Handles caching and processing.

        Args:
            df (pd.DataFrame): Input dataframe containing 'id', 'cell_type', 'source'.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: Dataframe with Multi-View features.
        """
        # Determine cache path
        if split == "train":
            cache_path = Config.CACHE_TRAIN_FEATURES
        elif split == "val":
            cache_path = Config.CACHE_VAL_FEATURES
        elif split == "test":
            cache_path = Config.CACHE_TEST_FEATURES
        else:
            raise ValueError(f"Unknown split: {split}")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} features from cache: {cache_path}")
            try:
                cached_df = pd.read_parquet(cache_path)
                # If in debug mode, we might need to filter the cached features
                # to match the input df IDs (if the input df is subsampled)
                input_ids = set(df["id"].unique())
                cached_ids = set(cached_df["id"].unique())

                # If cache is a superset or match, we filter and return
                if input_ids.issubset(cached_ids):
                    return cached_df[cached_df["id"].isin(input_ids)].reset_index(
                        drop=True
                    )
                else:
                    print("Cache missing some IDs. Reprocessing...")
            except Exception as e:
                print(f"Error loading cache: {e}. Reprocessing...")

        # 2. Process from scratch
        print(f"Generating Multi-View features for {split}...")

        # Ensure vectorizer is fitted (usually fitted on train before calling this)
        # If not fitted, fit on current df (fallback, though ideally fit on train)
        if self.vectorizer.tfidf is None:
            print("Vectorizer not initialized. Fitting on provided dataframe...")
            self.vectorizer.fit(df["source"])

        # Transform all text at once for efficiency
        print("Transforming text to vectors...")
        sparse_matrix, dense_matrix = self.vectorizer.transform(df["source"])

        # Group by notebook
        # We need to pass the corresponding slice of the matrices to each group
        # Since groupby shuffles order or creates copies, we need to be careful mapping indices.
        # Strategy: Iterate groups, use original indices to slice the global matrices.

        grouped = df.groupby("id", observed=True)
        all_features = []

        # Iterate groups
        # Note: We cannot use tqdm
        print(f"Processing {df['id'].nunique()} notebooks...")

        # Map global indices to groups
        # df.groupby preserves order of appearance if sort=False, but let's be robust.
        # We'll rely on the index of the dataframe.

        for nb_id, group in grouped:
            # Get integer locations (indices) of the group in the original df
            # This assumes df index is unique or we reset it.
            # Ideally, we pass the slice of matrices corresponding to the group rows.

            # To do this efficiently:
            # The matrix rows correspond to df rows.
            indices = group.index.values

            group_sparse = sparse_matrix[indices]
            group_dense = dense_matrix[indices]

            # Reset index of group for internal logic (0..N)
            group_reset = group.reset_index(drop=True)

            nb_feats = self._get_notebook_features(
                group_reset, group_sparse, group_dense
            )
            all_features.extend(nb_feats)

        # Create DataFrame
        feature_df = pd.DataFrame(all_features)

        # Optimize types
        for col in feature_df.columns:
            if col not in ["id", "cell_id"]:
                feature_df[col] = feature_df[col].astype(np.float32)

        # 3. Save to cache
        print(f"Saving {split} features to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feature_df.to_parquet(cache_path, index=False)

        return feature_df
