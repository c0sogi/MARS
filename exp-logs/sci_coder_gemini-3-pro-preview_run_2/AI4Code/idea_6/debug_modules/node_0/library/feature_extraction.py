import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class FeaturePipeline:
    """
    Manages feature extraction for the Stacked Hybrid Linear-Tree Ranking pipeline.
    Handles TF-IDF vectorization, LSA dimensionality reduction, and metadata extraction.
    """

    def __init__(self):
        # Level 1: Markdown TF-IDF
        self.md_tfidf = TfidfVectorizer(**Config.MD_TFIDF_PARAMS)

        # Level 2: Markdown LSA
        self.md_svd = TruncatedSVD(
            n_components=Config.MD_SVD_COMPONENTS, random_state=Config.SEED
        )

        # Level 2: Code Context Pipeline
        self.code_tfidf = TfidfVectorizer(**Config.CODE_TFIDF_PARAMS)
        self.code_svd = TruncatedSVD(
            n_components=Config.CODE_SVD_COMPONENTS, random_state=Config.SEED
        )

    def fit(self, df_md, df_nb, load_cached_data=True):
        """
        Fits the feature extraction pipelines (TF-IDF and SVD) or loads them from cache.

        Args:
            df_md (pd.DataFrame): DataFrame containing markdown cells.
            df_nb (pd.DataFrame): DataFrame containing notebook-level code context.
            load_cached_data (bool): Whether to attempt loading fitted models from disk.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Check if all artifacts exist
        artifacts = [
            Config.VECTORIZER_MD_PATH,
            Config.SVD_MD_PATH,
            Config.VECTORIZER_CODE_PATH,
            Config.SVD_CODE_PATH,
        ]

        all_exist = all(os.path.exists(p) for p in artifacts)

        if load_cached_data and all_exist:
            print("Loading fitted feature extractors from cache...")
            self.md_tfidf = joblib.load(Config.VECTORIZER_MD_PATH)
            self.md_svd = joblib.load(Config.SVD_MD_PATH)
            self.code_tfidf = joblib.load(Config.VECTORIZER_CODE_PATH)
            self.code_svd = joblib.load(Config.SVD_CODE_PATH)
            return

        print("Fitting feature extractors from scratch...")

        # 1. Fit Level 1 Markdown TF-IDF
        print("Fitting Markdown TF-IDF...")
        # Fill NaNs with empty string to prevent errors
        md_text = df_md["source"].fillna("").astype(str)
        md_sparse = self.md_tfidf.fit_transform(md_text)

        # 2. Fit Level 2 Markdown SVD (LSA)
        print("Fitting Markdown SVD...")
        self.md_svd.fit(md_sparse)

        # 3. Fit Level 2 Code Context Pipeline
        print("Fitting Code Context Pipeline...")
        code_text = df_nb["code_context"].fillna("").astype(str)
        code_sparse = self.code_tfidf.fit_transform(code_text)
        self.code_svd.fit(code_sparse)

        # Save models
        print("Saving fitted models to cache...")
        joblib.dump(self.md_tfidf, Config.VECTORIZER_MD_PATH)
        joblib.dump(self.md_svd, Config.SVD_MD_PATH)
        joblib.dump(self.code_tfidf, Config.VECTORIZER_CODE_PATH)
        joblib.dump(self.code_svd, Config.SVD_CODE_PATH)

    def transform_level1(self, df_md):
        """
        Transforms markdown text into sparse TF-IDF vectors for the Ridge model.

        Args:
            df_md (pd.DataFrame): DataFrame containing markdown cells.

        Returns:
            scipy.sparse.csr_matrix: Sparse feature matrix.
        """
        md_text = df_md["source"].fillna("").astype(str)
        return self.md_tfidf.transform(md_text)

    def transform_level2(self, df_md, df_nb, level1_preds=None):
        """
        Creates the dense feature matrix for the Level 2 Gradient Boosting model.
        Combines LSA semantics, code context, metadata, and optional Level 1 predictions.

        Args:
            df_md (pd.DataFrame): DataFrame containing markdown cells.
            df_nb (pd.DataFrame): DataFrame containing notebook-level info.
            level1_preds (np.array or list, optional): Predictions from the Ridge model.

        Returns:
            pd.DataFrame: Dense feature matrix.
        """
        print("Generating Level 2 features...")

        # --- 1. Markdown Semantic Features (LSA) ---
        md_text = df_md["source"].fillna("").astype(str)
        md_sparse = self.md_tfidf.transform(md_text)
        md_lsa = self.md_svd.transform(md_sparse)

        # Create DataFrame for LSA features
        lsa_cols = [f"md_lsa_{i}" for i in range(Config.MD_SVD_COMPONENTS)]
        df_features = pd.DataFrame(md_lsa, columns=lsa_cols, index=df_md.index)

        # --- 2. Code Context Features (LSA) ---
        # Transform notebook code contexts
        code_text = df_nb["code_context"].fillna("").astype(str)
        code_sparse = self.code_tfidf.transform(code_text)
        code_lsa = self.code_svd.transform(code_sparse)

        # Create a mapping from notebook_id to code LSA vector
        code_lsa_cols = [f"code_lsa_{i}" for i in range(Config.CODE_SVD_COMPONENTS)]
        df_code_lsa = pd.DataFrame(code_lsa, columns=code_lsa_cols)
        df_code_lsa["id"] = df_nb["id"].values

        # Merge code context features into the cell-level dataframe
        # We temporarily add 'id' to df_features for the merge, then remove it
        df_features["id"] = df_md["id"].values
        df_features = df_features.merge(df_code_lsa, on="id", how="left")

        # --- 3. Structural Metadata ---
        # Character length of the markdown cell
        df_features["char_len"] = df_md["source"].str.len().fillna(0)

        # Notebook-level metadata from df_nb
        # We already merged on 'id', so we can bring in total_cells via merge as well
        df_nb_meta = df_nb[["id", "total_cells"]].copy()
        df_features = df_features.merge(df_nb_meta, on="id", how="left")

        # Markdown Ratio (Density)
        # Calculate number of markdown cells per notebook based on the current df_md
        md_counts = df_md.groupby("id").size().reset_index(name="n_md")
        df_features = df_features.merge(md_counts, on="id", how="left")

        # Calculate ratio (handle division by zero if total_cells is 0, though unlikely)
        df_features["md_ratio"] = df_features["n_md"] / df_features["total_cells"]
        df_features["md_ratio"] = df_features["md_ratio"].fillna(0)

        # --- 4. Stacking (Level 1 Predictions) ---
        if level1_preds is not None:
            df_features["pred_ridge"] = level1_preds

        # Drop non-feature columns
        drop_cols = ["id", "n_md"]
        df_features = df_features.drop(
            columns=[c for c in drop_cols if c in df_features.columns]
        )

        return df_features
