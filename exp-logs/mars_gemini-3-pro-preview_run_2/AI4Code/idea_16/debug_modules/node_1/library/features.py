import os
import re
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Set, Optional

from library.config import Config
from library.utils import get_logger, Timer, seed_everything


class FeaturePipeline:
    """
    Implements the Multi-View Distributional Anchoring feature engineering pipeline.
    Manages Vectorization, Stage 1 Ridge Regression, and Stage 2 Distributional Features.
    """

    def __init__(self):
        self.logger = get_logger("FeaturePipeline")
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        # Paths for saving/loading models
        self.tfidf_path = os.path.join(self.working_dir, "tfidf_vectorizer.joblib")
        self.svd_path = os.path.join(self.working_dir, "svd_model.joblib")
        self.ridge_path = os.path.join(self.working_dir, "ridge_model.joblib")

        # Models
        self.tfidf = None
        self.svd = None
        self.ridge = None

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        return str(text).lower().strip()

    def extract_symbols(self, text: str) -> Set[str]:
        """Extracts variable and function names using regex."""
        # Regex for identifiers: starts with letter/underscore, followed by alphanumerics
        return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text))

    def _fit_vectorizers(self, texts: List[str]):
        """Fits TF-IDF and SVD on the provided corpus."""
        self.logger.info("Fitting TF-IDF Vectorizer...")
        self.tfidf = TfidfVectorizer(
            max_features=Config.MD_VOCAB_SIZE,
            ngram_range=Config.NGRAM_RANGE,
            sublinear_tf=True,
            strip_accents=None,  # As per lessons
            token_pattern=r"(?u)\b\w\w+\b",
        )
        tfidf_matrix = self.tfidf.fit_transform(texts)

        self.logger.info(
            f"Fitting TruncatedSVD ({Config.SVD_COMPONENTS} components)..."
        )
        self.svd = TruncatedSVD(
            n_components=Config.SVD_COMPONENTS, random_state=Config.RANDOM_STATE
        )
        self.svd.fit(tfidf_matrix)

        # Save models
        joblib.dump(self.tfidf, self.tfidf_path)
        joblib.dump(self.svd, self.svd_path)

    def _load_vectorizers(self):
        """Loads vectorizers from disk."""
        if os.path.exists(self.tfidf_path) and os.path.exists(self.svd_path):
            self.tfidf = joblib.load(self.tfidf_path)
            self.svd = joblib.load(self.svd_path)
        else:
            raise FileNotFoundError("Vectorizers not found. Run fit_pipeline first.")

    def fit_pipeline(self, df_md_train: pd.DataFrame):
        """
        Fits the initial vectorizers and the global Ridge model.
        Args:
            df_md_train: Training markdown DataFrame.
        """
        with Timer("Fit Pipeline"):
            # 1. Fit Vectorizers on training markdown
            self._fit_vectorizers(df_md_train["source"].apply(self.clean_text).tolist())

            # 2. Train Global Ridge Model (for inference use)
            self.logger.info("Training Global Ridge Model...")
            X = self.tfidf.transform(df_md_train["source"].apply(self.clean_text))
            y = df_md_train["rank"].values

            self.ridge = Ridge(
                alpha=Config.RIDGE_ALPHA, random_state=Config.RANDOM_STATE
            )
            self.ridge.fit(X, y)
            joblib.dump(self.ridge, self.ridge_path)

    def _get_ridge_predictions(
        self, df_md: pd.DataFrame, split_name: str
    ) -> np.ndarray:
        """
        Generates Stage 1 predictions.
        - For 'train': Uses K-Fold OOF predictions.
        - For 'val'/'test': Uses the global fitted Ridge model.
        """
        texts = df_md["source"].apply(self.clean_text)
        X = self.tfidf.transform(texts)

        if split_name == "train":
            self.logger.info("Generating OOF Ridge predictions for training set...")
            y = df_md["rank"].values
            oof_preds = np.zeros(len(df_md))
            kf = KFold(
                n_splits=Config.NUM_FOLDS,
                shuffle=True,
                random_state=Config.RANDOM_STATE,
            )

            # We need to perform CV on the sparse matrix X
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr = y[train_idx]

                model = Ridge(
                    alpha=Config.RIDGE_ALPHA, random_state=Config.RANDOM_STATE
                )
                model.fit(X_tr, y_tr)
                oof_preds[val_idx] = model.predict(X_val)

            return oof_preds
        else:
            # Load global model if not in memory
            if self.ridge is None:
                if os.path.exists(self.ridge_path):
                    self.ridge = joblib.load(self.ridge_path)
                else:
                    raise FileNotFoundError(
                        "Ridge model not found. Fit pipeline first."
                    )
            return self.ridge.predict(X)

    def _compute_distributional_features(
        self, df_md: pd.DataFrame, df_nb: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes Multi-View Distributional Anchoring features.
        """
        self.logger.info("Computing Distributional Anchor Features...")

        # 1. Pre-compute Code Vectors per Notebook
        # Flatten code cells to vectorize in batch
        nb_code_map = {}  # nb_id -> {indices in flat list}
        flat_code_sources = []
        flat_code_nb_ids = []

        # Also prepare symbolic sets
        nb_code_symbols = {}  # nb_id -> list of sets

        for _, row in df_nb.iterrows():
            nb_id = row["notebook_id"]
            code_srcs = row["code_sources"]

            start_idx = len(flat_code_sources)
            flat_code_sources.extend([self.clean_text(c) for c in code_srcs])
            end_idx = len(flat_code_sources)

            nb_code_map[nb_id] = (start_idx, end_idx)

            # Symbolic
            nb_code_symbols[nb_id] = [self.extract_symbols(c) for c in code_srcs]

        # Vectorize all code cells
        self.logger.info(f"Vectorizing {len(flat_code_sources)} code cells...")
        # Batch transform to avoid memory issues if too large, but 140k nbs * ~30 cells ~ 4.2M cells.
        # Sparse matrix of 4.2M x 60k is manageable in RAM usually (~2GB).
        code_tfidf = self.tfidf.transform(flat_code_sources)
        code_svd = self.svd.transform(code_tfidf)

        # Vectorize all markdown cells
        md_texts = df_md["source"].apply(self.clean_text).tolist()
        md_tfidf = self.tfidf.transform(md_texts)
        md_svd = self.svd.transform(md_tfidf)
        md_symbols = [self.extract_symbols(t) for t in md_texts]

        # 2. Compute Features per Notebook
        # We group MD cells by notebook to process one notebook at a time
        df_md["original_index"] = range(len(df_md))
        grouped = df_md.groupby("notebook_id")

        # Feature containers
        n_bins = Config.N_BINS
        n_md = len(df_md)

        # Features:
        # [Lexical Hist (10)] + [Latent Hist (10)] + [Symbolic Hist (10)] + [TopK Mean, TopK Std]
        # Total 32 features
        feats_lex = np.zeros((n_md, n_bins), dtype=np.float32)
        feats_lat = np.zeros((n_md, n_bins), dtype=np.float32)
        feats_sym = np.zeros((n_md, n_bins), dtype=np.float32)
        feats_topk = np.zeros((n_md, 2), dtype=np.float32)

        # Iterate over notebooks
        # Using tqdm for progress
        for nb_id, group in tqdm(
            grouped, total=len(grouped), desc="Processing Notebooks"
        ):
            if nb_id not in nb_code_map:
                continue

            # Get Code Data for this notebook
            c_start, c_end = nb_code_map[nb_id]
            n_code = c_end - c_start
            if n_code == 0:
                continue

            # Code Vectors
            c_vec_tfidf = code_tfidf[c_start:c_end]  # Sparse
            c_vec_svd = code_svd[c_start:c_end]  # Dense
            c_syms = nb_code_symbols[nb_id]  # List of sets

            # Normalized Ranks of code cells (0.0 to 1.0)
            # If n_code=1, rank is 0.0. If n_code > 1, linspace.
            if n_code > 1:
                c_ranks = np.linspace(0, 1, n_code)
            else:
                c_ranks = np.array([0.0])

            # Get MD Data for this notebook
            md_indices = group["original_index"].values
            m_vec_tfidf = md_tfidf[md_indices]  # Sparse
            m_vec_svd = md_svd[md_indices]  # Dense

            # --- Lexical Similarity (TF-IDF) ---
            # Sparse dot product
            sim_lex = cosine_similarity(
                m_vec_tfidf, c_vec_tfidf
            )  # (n_md_in_nb, n_code)

            # --- Latent Similarity (SVD) ---
            sim_lat = cosine_similarity(m_vec_svd, c_vec_svd)  # (n_md_in_nb, n_code)

            # --- Symbolic Similarity (Jaccard) ---
            # Custom loop as Jaccard isn't matrix optimized easily for sets
            sim_sym = np.zeros((len(md_indices), n_code), dtype=np.float32)
            for i, md_idx in enumerate(md_indices):
                m_s = md_symbols[md_idx]
                if not m_s:
                    continue
                for j in range(n_code):
                    c_s = c_syms[j]
                    if not c_s:
                        continue
                    intersection = len(m_s & c_s)
                    union = len(m_s | c_s)
                    if union > 0:
                        sim_sym[i, j] = intersection / union

            # --- Aggregation (Binning & TopK) ---
            # Bin indices for code cells
            bin_edges = np.linspace(0, 1, n_bins + 1)
            # digitize returns 1..N_BINS, we want 0..N_BINS-1
            # We clamp to ensure 1.0 goes to last bin
            c_bin_indices = np.digitize(c_ranks, bin_edges) - 1
            c_bin_indices = np.clip(c_bin_indices, 0, n_bins - 1)

            for i in range(len(md_indices)):
                row_idx = md_indices[i]

                # 1. Histograms
                # Sum similarities per bin
                np.add.at(feats_lex[row_idx], c_bin_indices, sim_lex[i])
                np.add.at(feats_lat[row_idx], c_bin_indices, sim_lat[i])
                np.add.at(feats_sym[row_idx], c_bin_indices, sim_sym[i])

                # Normalize histograms (sum to 1)
                sum_lex = feats_lex[row_idx].sum()
                if sum_lex > 1e-6:
                    feats_lex[row_idx] /= sum_lex

                sum_lat = feats_lat[row_idx].sum()
                if sum_lat > 1e-6:
                    feats_lat[row_idx] /= sum_lat

                sum_sym = feats_sym[row_idx].sum()
                if sum_sym > 1e-6:
                    feats_sym[row_idx] /= sum_sym

                # 2. Top-K Smoothing (Using Latent Sim as primary signal for neighbors)
                # Sort indices by similarity descending
                top_k_indices = np.argsort(sim_lat[i])[::-1][: Config.TOP_K]
                top_k_ranks = c_ranks[top_k_indices]

                feats_topk[row_idx, 0] = np.mean(top_k_ranks)
                feats_topk[row_idx, 1] = (
                    np.std(top_k_ranks) if len(top_k_ranks) > 1 else 0.0
                )

        # Construct DataFrame
        feature_cols = []
        feature_data = {}

        for b in range(n_bins):
            feature_data[f"dist_lex_{b}"] = feats_lex[:, b]
            feature_cols.append(f"dist_lex_{b}")
        for b in range(n_bins):
            feature_data[f"dist_lat_{b}"] = feats_lat[:, b]
            feature_cols.append(f"dist_lat_{b}")
        for b in range(n_bins):
            feature_data[f"dist_sym_{b}"] = feats_sym[:, b]
            feature_cols.append(f"dist_sym_{b}")

        feature_data["topk_mean"] = feats_topk[:, 0]
        feature_data["topk_std"] = feats_topk[:, 1]

        df_features = pd.DataFrame(feature_data, index=df_md.index)
        return df_features

    def transform_pipeline(
        self,
        df_md: pd.DataFrame,
        df_nb: pd.DataFrame,
        split_name: str,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Main method to generate features for a given split.
        Checks cache, generates if missing, and saves.
        """
        cache_path = os.path.join(self.working_dir, f"features_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        self.logger.info(f"Generating features for {split_name}...")

        # Ensure vectorizers are loaded
        if self.tfidf is None:
            self._load_vectorizers()

        # 1. Stage 1: Ridge Predictions
        ridge_preds = self._get_ridge_predictions(df_md, split_name)

        # 2. Stage 2: Distributional Anchors
        df_dist = self._compute_distributional_features(df_md, df_nb)

        # 3. Combine
        df_final = df_md[["notebook_id", "cell_id"]].copy()
        if "rank" in df_md.columns:
            df_final["target"] = df_md["rank"]

        df_final["pred_ridge"] = ridge_preds
        df_final = pd.concat([df_final, df_dist], axis=1)

        # 4. Metadata Features (Simple)
        # Add notebook length info
        nb_lens = df_nb.set_index("notebook_id")["code_sources"].apply(len).to_dict()
        df_final["n_code_cells"] = df_final["notebook_id"].map(nb_lens).fillna(0)

        # Save to cache
        self.logger.info(f"Saving features to {cache_path}")
        df_final.to_parquet(cache_path, index=False)

        return df_final
