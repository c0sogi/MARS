import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class AnchorEngine:
    """
    Implements the Multi-View Anchor mechanism.
    Calculates similarity-based features linking Markdown cells to Code cells
    using both explicit (Lexical/TF-IDF) and implicit (Latent/SVD) representations.
    """

    def __init__(self, semantic_space):
        """
        Args:
            semantic_space: An instance of library.vectorizers.SemanticSpace
                            with fitted TF-IDF and SVD models.
        """
        self.semantic_space = semantic_space

    def compute_features(self, df, cache_name, load_cached_data=True):
        """
        Computes anchor features for all markdown cells in the dataframe.

        Args:
            df (pd.DataFrame): DataFrame containing 'cell_id', 'cell_type', 'source', 'notebook_id'.
            cache_name (str): Identifier for the cache file (e.g., 'train', 'test').
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: DataFrame with columns ['cell_id', 'lexical_anchor_rank',
                          'lexical_anchor_sim', 'latent_anchor_rank', 'latent_anchor_sim'].
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(
            Config.WORKING_DIR, f"{cache_name}_anchor_features.parquet"
        )

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached anchor features from {cache_path}...")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print(f"Computing anchor features for {cache_name} set...")

        # 2. Global Vectorization
        # We transform all cells at once for efficiency, then slice by notebook later.
        # Note: We need vectors for both Code and Markdown to compute similarity.
        print("Generating global TF-IDF and SVD vectors...")

        # Ensure indices are aligned
        df = df.reset_index(drop=True)

        # Transform text content
        # We fill NaN with empty string to ensure vectorizer works
        text_content = df["source"].astype(str).fillna("")

        tfidf_matrix = self.semantic_space.transform_tfidf(text_content)
        svd_matrix = self.semantic_space.transform_svd(text_content)

        # 3. Process Notebooks in Parallel
        # We pass indices to the worker to slice the global matrices
        unique_nb_ids = df["notebook_id"].unique()

        # Create a mapping of notebook_id to dataframe indices
        # This is faster than repeated filtering inside the loop
        nb_groups = df.groupby("notebook_id", observed=True).indices

        print(
            f"Processing {len(unique_nb_ids)} notebooks with {Config.NUM_WORKERS} workers..."
        )

        results = Parallel(n_jobs=Config.NUM_WORKERS, backend="threading")(
            delayed(self._process_single_notebook)(
                nb_id, indices, df, tfidf_matrix, svd_matrix
            )
            for nb_id, indices in nb_groups.items()
        )

        # Flatten results
        flat_results = [item for sublist in results for item in sublist]

        # Create DataFrame
        features_df = pd.DataFrame(flat_results)

        # Ensure correct types
        cols = [
            "lexical_anchor_rank",
            "lexical_anchor_sim",
            "latent_anchor_rank",
            "latent_anchor_sim",
        ]
        for c in cols:
            if c in features_df.columns:
                features_df[c] = features_df[c].astype(np.float32)

        # 4. Save to cache
        print(f"Saving anchor features to {cache_path}...")
        features_df.to_parquet(cache_path, index=False)

        return features_df

    def _process_single_notebook(self, nb_id, indices, df, tfidf_matrix, svd_matrix):
        """
        Internal worker function to process a single notebook.
        Computes pairwise similarities between MD and Code cells.
        """
        # Get the subset of the dataframe for this notebook
        # We use the indices provided by groupby
        nb_df = df.iloc[indices]

        # Identify Code and Markdown indices relative to the global matrix
        # indices is a numpy array of global indices
        code_mask = (nb_df["cell_type"] == "code").values
        md_mask = (nb_df["cell_type"] == "markdown").values

        global_code_indices = indices[code_mask]
        global_md_indices = indices[md_mask]

        # If no markdown cells, nothing to return
        if len(global_md_indices) == 0:
            return []

        md_cell_ids = nb_df.loc[md_mask, "cell_id"].values

        # Default features if no code cells exist
        if len(global_code_indices) == 0:
            return [
                {
                    "cell_id": cid,
                    "lexical_anchor_rank": 0.0,
                    "lexical_anchor_sim": 0.0,
                    "latent_anchor_rank": 0.0,
                    "latent_anchor_sim": 0.0,
                }
                for cid in md_cell_ids
            ]

        # Calculate Code Ranks
        # The code cells are guaranteed to be in correct order in the dataframe
        n_code = len(global_code_indices)
        if n_code == 1:
            code_ranks = np.array([0.0])
        else:
            code_ranks = np.arange(n_code) / (n_code - 1)

        # --- Lexical View (TF-IDF) ---
        # Slice global sparse matrix
        # shape: (n_md, n_features) and (n_code, n_features)
        md_tfidf = tfidf_matrix[global_md_indices]
        code_tfidf = tfidf_matrix[global_code_indices]

        # Compute Cosine Similarity: (n_md, n_code)
        lex_sim_matrix = cosine_similarity(md_tfidf, code_tfidf)

        # Find best anchors
        lex_best_idx = np.argmax(lex_sim_matrix, axis=1)
        lex_best_sim = np.max(lex_sim_matrix, axis=1)
        lex_best_ranks = code_ranks[lex_best_idx]

        # --- Latent View (SVD) ---
        # Slice global dense matrix
        md_svd = svd_matrix[global_md_indices]
        code_svd = svd_matrix[global_code_indices]

        # Compute Cosine Similarity
        lat_sim_matrix = cosine_similarity(md_svd, code_svd)

        # Find best anchors
        lat_best_idx = np.argmax(lat_sim_matrix, axis=1)
        lat_best_sim = np.max(lat_sim_matrix, axis=1)
        lat_best_ranks = code_ranks[lat_best_idx]

        # Combine results
        results = []
        for i, cid in enumerate(md_cell_ids):
            results.append(
                {
                    "cell_id": cid,
                    "lexical_anchor_rank": lex_best_ranks[i],
                    "lexical_anchor_sim": lex_best_sim[i],
                    "latent_anchor_rank": lat_best_ranks[i],
                    "latent_anchor_sim": lat_best_sim[i],
                }
            )

        return results
