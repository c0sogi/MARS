import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from library.config import Config


class FeatureEngine:
    """
    Handles feature extraction for the Stacked Hybrid Ranking model.
    Implements Global Vectorization (TF-IDF, SVD) and Multi-View Instance-Based Neighborhood features.
    """

    def __init__(self):
        self.tfidf_path = os.path.join(Config.WORKING_DIR, "tfidf_vectorizer.joblib")
        self.svd_path = os.path.join(Config.WORKING_DIR, "svd_model.joblib")

        self.tfidf_model = None
        self.svd_model = None

    def fit_global_vectorizers(self, df_train, load_cached_models=True):
        """
        Fits TF-IDF and SVD models on the training markdown corpus.

        Args:
            df_train (pd.DataFrame): Training data containing 'source' and 'cell_type'.
            load_cached_models (bool): Whether to try loading models from disk.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Try loading from cache
        if (
            load_cached_models
            and os.path.exists(self.tfidf_path)
            and os.path.exists(self.svd_path)
        ):
            print("Loading global vectorizers from cache...")
            self.tfidf_model = joblib.load(self.tfidf_path)
            self.svd_model = joblib.load(self.svd_path)
            return

        # 2. Fit from scratch
        print("Fitting global vectorizers...")

        # Filter for markdown cells only for vocabulary building
        markdowns = (
            df_train[df_train["cell_type"] == "markdown"]["source"].astype(str).tolist()
        )

        # Initialize and fit TF-IDF
        self.tfidf_model = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE,
            ngram_range=Config.NGRAM_RANGE,
            strip_accents=Config.STRIP_ACCENTS,
            min_df=Config.MIN_DF,
            use_idf=Config.USE_IDF,
            sublinear_tf=Config.SUBLINEAR_TF,
        )
        tfidf_matrix = self.tfidf_model.fit_transform(markdowns)

        # Initialize and fit SVD
        self.svd_model = TruncatedSVD(
            n_components=Config.SVD_COMPONENTS, random_state=Config.SEED
        )
        self.svd_model.fit(tfidf_matrix)

        # 3. Save to cache
        print(f"Saving vectorizers to {Config.WORKING_DIR}...")
        joblib.dump(self.tfidf_model, self.tfidf_path)
        joblib.dump(self.svd_model, self.svd_path)

    def get_stage1_features(self, df):
        """
        Transforms text into sparse TF-IDF vectors for the Stage 1 Ridge Regressor.

        Args:
            df (pd.DataFrame): Dataframe containing 'source'.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF features.
        """
        if self.tfidf_model is None:
            raise ValueError(
                "TF-IDF model not fitted. Call fit_global_vectorizers first."
            )

        return self.tfidf_model.transform(df["source"].astype(str))

    def _calculate_jaccard_similarity(self, set_a, list_sets_b):
        """
        Computes Jaccard similarity between a set and a list of sets.
        """
        if not set_a:
            return np.zeros(len(list_sets_b))

        scores = []
        for set_b in list_sets_b:
            if not set_b:
                scores.append(0.0)
                continue
            intersection = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))
            scores.append(intersection / union if union > 0 else 0.0)
        return np.array(scores)

    def _process_notebook_group(self, nb_id, group_df, tfidf_mat, svd_mat):
        """
        Process a single notebook to extract Multi-View Instance features.
        Designed to be run in parallel.
        """
        # Separate Code and Markdown
        code_mask = group_df["cell_type"] == "code"
        md_mask = group_df["cell_type"] == "markdown"

        code_df = group_df[code_mask]
        md_df = group_df[md_mask]

        if len(code_df) == 0 or len(md_df) == 0:
            # Edge case: No code or no markdown. Return empty or default features.
            # We return features for markdown cells filled with defaults.
            features = []
            for _, row in md_df.iterrows():
                feat_row = {"id": nb_id, "cell_id": row["cell_id"]}
                for view in ["lexical", "latent", "symbolic"]:
                    for k in range(1, 4):
                        feat_row[f"{view}_neighbor_{k}_rank"] = 0.5
                        feat_row[f"{view}_neighbor_{k}_score"] = 0.0
                    feat_row[f"{view}_top5_mean"] = 0.5
                    feat_row[f"{view}_top5_std"] = 0.0
                features.append(feat_row)
            return features

        # Establish Code Skeleton Ranks
        # In both Train and Test, we rely on the order provided in the dataframe.
        # For Train, this matches 'rank'. For Test, it matches file order (which is correct for code).
        n_code = len(code_df)
        code_ranks = np.linspace(0, 1, n_code)

        # Get Vectors
        # We need relative indices within the passed matrices
        # The matrices passed to this function are slices corresponding to this group
        code_indices = np.where(code_mask)[0]
        md_indices = np.where(md_mask)[0]

        code_tfidf = tfidf_mat[code_indices]
        md_tfidf = tfidf_mat[md_indices]

        code_svd = svd_mat[code_indices]
        md_svd = svd_mat[md_indices]

        # Prepare Symbolic Sets
        code_symbols = [set(s.split()) if s else set() for s in code_df["symbols"]]
        md_symbols = [set(s.split()) if s else set() for s in md_df["symbols"]]

        # Compute Pairwise Similarities
        # 1. Lexical (TF-IDF Cosine)
        sim_lexical = cosine_similarity(md_tfidf, code_tfidf)

        # 2. Latent (SVD Cosine)
        sim_latent = cosine_similarity(md_svd, code_svd)

        # 3. Symbolic (Jaccard)
        # Computed manually per markdown cell

        results = []

        # Iterate over markdown cells to extract features
        for i, (idx, row) in enumerate(md_df.iterrows()):
            feat_row = {"id": nb_id, "cell_id": row["cell_id"]}

            # --- View Processing ---
            views_data = {
                "lexical": sim_lexical[i],
                "latent": sim_latent[i],
                "symbolic": self._calculate_jaccard_similarity(
                    md_symbols[i], code_symbols
                ),
            }

            for view_name, scores in views_data.items():
                # Find Top-K neighbors
                # Argsort returns indices of scores sorted ascending, so we take tail and reverse
                top_k_indices = np.argsort(scores)[-Config.NUM_NEIGHBORS :][::-1]

                # Extract features for Top 1, 2, 3
                for k in range(1, 4):
                    if k <= len(top_k_indices):
                        neighbor_idx = top_k_indices[k - 1]
                        feat_row[f"{view_name}_neighbor_{k}_rank"] = code_ranks[
                            neighbor_idx
                        ]
                        feat_row[f"{view_name}_neighbor_{k}_score"] = scores[
                            neighbor_idx
                        ]
                    else:
                        # Fallback if fewer than k neighbors (e.g. only 1 code cell)
                        feat_row[f"{view_name}_neighbor_{k}_rank"] = 0.5
                        feat_row[f"{view_name}_neighbor_{k}_score"] = 0.0

                # Aggregate stats for Top 5 (or fewer)
                top_ranks = code_ranks[top_k_indices]
                feat_row[f"{view_name}_top5_mean"] = np.mean(top_ranks)
                feat_row[f"{view_name}_top5_std"] = (
                    np.std(top_ranks) if len(top_ranks) > 1 else 0.0
                )

            results.append(feat_row)

        return results

    def get_stage2_features(self, df, split="train", load_cached_data=True):
        """
        Generates Multi-View Instance-Based features for Stage 2.

        Args:
            df (pd.DataFrame): Processed dataframe with 'source', 'symbols', 'cell_type'.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached parquet file.

        Returns:
            pd.DataFrame: Dataframe with instance features.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(
            Config.WORKING_DIR, f"stage2_features_{split}.parquet"
        )

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Checking cache: {cache_path}")
            try:
                df_cache = pd.read_parquet(cache_path)

                # Validate Cache Consistency (Cite debug_lesson_1)
                expected_ids = set(df["id"].unique())
                cached_ids = set(df_cache["id"].unique())

                if expected_ids == cached_ids:
                    print(
                        f"Cache valid. Loading Stage 2 features for {split} from cache."
                    )
                    return df_cache
                else:
                    print(
                        f"Cache mismatch! Input has {len(expected_ids)} notebooks, cache has {len(cached_ids)}."
                    )
                    print("Invalidating cache and reprocessing...")
            except Exception as e:
                print(f"Failed to read or validate cache: {e}. Reprocessing...")

        print(f"Generating Stage 2 features for {split}...")

        if self.tfidf_model is None or self.svd_model is None:
            raise ValueError(
                "Vectorizers not fitted. Call fit_global_vectorizers first."
            )

        # 2. Pre-compute Global Vectors
        # We transform the entire dataframe at once for efficiency
        print("Transforming text to vectors...")
        tfidf_full = self.tfidf_model.transform(df["source"].astype(str))
        svd_full = self.svd_model.transform(tfidf_full)

        # 3. Parallel Processing by Notebook
        # Group by ID to process each notebook independently
        # We need to map dataframe indices to matrix indices
        # Since df index might not be reset, we use integer location

        # Create a list of (nb_id, group_df, tfidf_slice, svd_slice) tuples
        tasks = []

        # It's faster to iterate via groupby if we can slice the sparse matrix efficiently.
        # However, sparse slicing is fast.
        # We reset index to ensure alignment with tfidf_full rows 0..N
        df_reset = df.reset_index(drop=True)

        # Group indices
        groups = df_reset.groupby("id").indices

        print(
            f"Processing {len(groups)} notebooks with {Config.NUM_WORKERS} workers..."
        )

        # Define helper to unpack arguments
        def process_wrapper(nb_id, indices):
            group_df = df_reset.iloc[indices]
            # Slice matrices
            tfidf_slice = tfidf_full[indices]
            svd_slice = svd_full[indices]
            return self._process_notebook_group(nb_id, group_df, tfidf_slice, svd_slice)

        # Execute Parallel Loop
        # require='sharedmem' avoids pickling large matrices, using threads
        results_nested = Parallel(n_jobs=Config.NUM_WORKERS, require="sharedmem")(
            delayed(process_wrapper)(nb_id, idxs)
            for nb_id, idxs in tqdm(groups.items(), disable=True)
        )

        # Flatten results
        flat_results = [item for sublist in results_nested for item in sublist]
        feature_df = pd.DataFrame(flat_results)

        # 4. Save to Cache
        print(f"Saving Stage 2 features to {cache_path}...")
        feature_df.to_parquet(cache_path, index=False)

        return feature_df
