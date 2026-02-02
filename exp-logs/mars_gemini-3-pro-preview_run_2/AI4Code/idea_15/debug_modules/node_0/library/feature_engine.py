import os
import numpy as np
import pandas as pd
import joblib
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Set

from library.config import Config
from library.utils import preprocess_text, extract_symbolic_tokens


class FeatureEngine:
    """
    Handles feature engineering for the notebook cell ordering task.
    Implements Multi-View Anchors (Lexical, Latent, Symbolic) and Text Vectorization.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.tfidf_path = os.path.join(self.working_dir, "tfidf_vectorizer.joblib")
        self.svd_path = os.path.join(self.working_dir, "svd_model.joblib")

        # Models
        self.tfidf = None
        self.svd = None

    def _load_models(self):
        """Loads vectorizers from disk if they exist."""
        if os.path.exists(self.tfidf_path):
            self.tfidf = joblib.load(self.tfidf_path)
        if os.path.exists(self.svd_path):
            self.svd = joblib.load(self.svd_path)

    def fit(self, df: pd.DataFrame):
        """
        Fits the TF-IDF and SVD models on the provided dataframe (usually training set).
        Saves the models to disk.
        """
        print("Fitting TextVectorizationPipeline...")
        # Preprocess text
        text_corpus = df["source"].apply(preprocess_text).fillna("").tolist()

        # 1. TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE,
            ngram_range=Config.NGRAM_RANGE,
            min_df=Config.MIN_DF,
            use_idf=Config.USE_IDF,
            sublinear_tf=Config.SUBLINEAR_TF,
            strip_accents=Config.STRIP_ACCENTS,
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        tfidf_matrix = self.tfidf.fit_transform(text_corpus)
        joblib.dump(self.tfidf, self.tfidf_path)
        print(f"TF-IDF fitted and saved. Shape: {tfidf_matrix.shape}")

        # 2. SVD
        self.svd = TruncatedSVD(
            n_components=Config.SVD_COMPONENTS,
            n_iter=Config.SVD_N_ITER,
            random_state=Config.RANDOM_STATE,
        )
        self.svd.fit(tfidf_matrix)
        joblib.dump(self.svd, self.svd_path)
        print(
            f"SVD fitted and saved. Explained Variance: {self.svd.explained_variance_ratio_.sum():.4f}"
        )

    def _compute_notebook_anchors(
        self,
        nb_id: str,
        group_df: pd.DataFrame,
        tfidf_matrix_sub: scipy.sparse.csr_matrix,
        svd_matrix_sub: np.ndarray,
        symbolic_sets_sub: List[Set[str]],
    ) -> List[Dict]:
        """
        Worker function to compute anchor features for a single notebook.
        """
        # Separate Code and Markdown
        # Reset index to be relative to the notebook group (0 to N-1)
        group_df = group_df.reset_index(drop=True)

        code_mask = group_df["cell_type"] == "code"
        md_mask = group_df["cell_type"] == "markdown"

        code_indices = group_df.index[code_mask].tolist()
        md_indices = group_df.index[md_mask].tolist()

        if not code_indices:
            # Edge case: No code cells. Return default features for MD cells.
            results = []
            for idx in md_indices:
                row = group_df.iloc[idx]
                feat = {
                    "id": nb_id,
                    "cell_id": row["cell_id"],
                    "rank": row["rank"],
                    "md_ratio": 1.0,
                    "total_code": 0,
                }
                # Add zeroed features
                for prefix in ["lex", "lat", "sym"]:
                    feat[f"{prefix}_mean"] = 0.5
                    feat[f"{prefix}_std"] = 0.0

                # Add SVD features
                for i in range(Config.SVD_COMPONENTS):
                    feat[f"svd_{i}"] = svd_matrix_sub[idx, i]

                results.append(feat)
            return results

        # Assign ranks to code cells (0.0 to 1.0)
        n_code = len(code_indices)
        code_ranks = np.linspace(0, 1, n_code) if n_code > 1 else np.array([0.0])

        # Prepare Code Features for Similarity
        # Slice the notebook-level matrices using local indices
        code_tfidf = tfidf_matrix_sub[code_indices]
        code_svd = svd_matrix_sub[code_indices]
        code_sym = [symbolic_sets_sub[i] for i in code_indices]

        results = []

        # Iterate over Markdown cells
        for md_idx in md_indices:
            md_row = group_df.iloc[md_idx]

            # --- 1. Lexical View (TF-IDF) ---
            # Compute cosine similarity between this MD cell and all Code cells
            md_vec_tfidf = tfidf_matrix_sub[md_idx]
            # sparse dot product
            lex_sims = code_tfidf.dot(md_vec_tfidf.T).toarray().flatten()

            # --- 2. Latent View (SVD) ---
            md_vec_svd = svd_matrix_sub[md_idx].reshape(1, -1)
            lat_sims = cosine_similarity(md_vec_svd, code_svd).flatten()

            # --- 3. Symbolic View (Jaccard) ---
            md_sym = symbolic_sets_sub[md_idx]
            sym_sims = []
            if not md_sym:
                sym_sims = np.zeros(n_code)
            else:
                for c_sym in code_sym:
                    if not c_sym:
                        sym_sims.append(0.0)
                    else:
                        intersection = len(md_sym & c_sym)
                        union = len(md_sym | c_sym)
                        sym_sims.append(intersection / union)
                sym_sims = np.array(sym_sims)

            # --- Aggregate Anchors ---
            feats = {
                "id": nb_id,
                "cell_id": md_row["cell_id"],
                "rank": md_row["rank"],
                "md_ratio": len(md_indices) / len(group_df),
                "total_code": n_code,
            }

            # Helper to compute stats for top K
            def get_stats(sim_scores, ranks):
                # Get indices of top K scores
                # argsort is ascending, so take tail and reverse
                k = min(Config.TOP_K, len(sim_scores))
                top_k_idx = np.argsort(sim_scores)[-k:][::-1]
                top_k_ranks = ranks[top_k_idx]

                # Weighted mean/std? Prompt says "Mean Rank and Std Dev"
                # We can just use simple stats of the ranks of the neighbors
                return np.mean(top_k_ranks), np.std(top_k_ranks)

            l_mean, l_std = get_stats(lex_sims, code_ranks)
            feats["lex_mean"] = l_mean
            feats["lex_std"] = l_std

            lat_mean, lat_std = get_stats(lat_sims, code_ranks)
            feats["lat_mean"] = lat_mean
            feats["lat_std"] = lat_std

            sym_mean, sym_std = get_stats(sym_sims, code_ranks)
            feats["sym_mean"] = sym_mean
            feats["sym_std"] = sym_std

            # Add SVD vector components
            for i in range(Config.SVD_COMPONENTS):
                feats[f"svd_{i}"] = svd_matrix_sub[md_idx, i]

            results.append(feats)

        return results

    def transform(
        self, df: pd.DataFrame, name: str = "train", load_cached_data: bool = True
    ) -> Tuple[scipy.sparse.csr_matrix, pd.DataFrame]:
        """
        Transforms the input dataframe into features.
        Returns:
            1. Sparse TF-IDF matrix (for Ridge/Stage 1) - aligned with the returned DataFrame rows.
            2. Dense DataFrame (for LGBM/Stage 2) - containing anchors, metadata, and SVD features.
        """
        # Define cache paths
        cache_sparse = os.path.join(self.working_dir, f"{name}_tfidf.npz")
        cache_dense = os.path.join(self.working_dir, f"{name}_features.parquet")

        # 1. Try loading from cache
        if (
            load_cached_data
            and os.path.exists(cache_sparse)
            and os.path.exists(cache_dense)
        ):
            # print(f"Loading cached features for {name}...")
            tfidf_matrix = scipy.sparse.load_npz(cache_sparse)
            df_features = pd.read_parquet(cache_dense)
            return tfidf_matrix, df_features

        # 2. Compute from scratch
        print(f"Generating features for {name}...")
        self._load_models()
        if self.tfidf is None or self.svd is None:
            raise ValueError(
                "Models not fitted. Call fit() first or ensure models are in working dir."
            )

        # Preprocess text
        # Ensure we are working with strings
        df["source"] = df["source"].fillna("").astype(str)
        text_corpus = df["source"].apply(preprocess_text).tolist()

        # Transform global matrices
        print("Transforming TF-IDF and SVD...")
        tfidf_full = self.tfidf.transform(text_corpus)
        svd_full = self.svd.transform(tfidf_full)

        # Extract Symbolic Tokens
        print("Extracting symbolic tokens...")
        symbolic_tokens = [set(extract_symbolic_tokens(txt)) for txt in df["source"]]

        # Group by notebook for anchor calculation
        # We need to map global indices to notebook groups
        # We can add a 'global_idx' col to df to track this
        df_w_idx = df.copy()
        df_w_idx["global_idx"] = np.arange(len(df))
        grouped = df_w_idx.groupby("id")

        # Define parallel worker
        def worker(nb_id, group):
            g_indices = group["global_idx"].values

            # Slice global data
            # Note: Slicing CSR by rows is relatively efficient
            sub_tfidf = tfidf_full[g_indices]
            sub_svd = svd_full[g_indices]
            sub_sym = [symbolic_tokens[i] for i in g_indices]

            return self._compute_notebook_anchors(
                nb_id, group, sub_tfidf, sub_svd, sub_sym
            )

        print("Computing Multi-View Anchors (Parallel)...")
        # Use 'threading' backend to share memory of tfidf_full/svd_full without pickling
        results_list = Parallel(n_jobs=Config.NUM_WORKERS, backend="threading")(
            delayed(worker)(nb_id, group) for nb_id, group in grouped
        )

        # Flatten results
        flat_results = [item for sublist in results_list for item in sublist]
        df_features = pd.DataFrame(flat_results)

        # Filter the sparse TF-IDF matrix to return only rows corresponding to the returned DataFrame
        # The returned DataFrame contains only Markdown cells (usually).
        # We need to align the sparse matrix output with df_features.

        # Re-construct the alignment
        # The df_features has 'id' and 'cell_id'. We can join with df_w_idx to get global_idx.
        # This ensures the sparse matrix rows match the dense dataframe rows 1:1.
        df_features_merged = df_features.merge(
            df_w_idx[["id", "cell_id", "global_idx"]], on=["id", "cell_id"], how="left"
        )

        relevant_indices = df_features_merged["global_idx"].values
        tfidf_out = tfidf_full[relevant_indices]

        # Drop auxiliary columns from dense features
        df_features_final = df_features_merged.drop(columns=["global_idx"])

        # Cache results
        print(f"Caching features to {self.working_dir}...")
        scipy.sparse.save_npz(cache_sparse, tfidf_out)
        df_features_final.to_parquet(cache_dense, index=False)

        return tfidf_out, df_features_final
