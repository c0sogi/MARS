import os
import ast
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import nltk

# Try to import VADER, handle if not available/downloadable
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

from library.config import (
    CACHE_DIR,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    SBERT_MODEL_NAME,
    EMBEDDING_DIM,
    MAX_TEXT_LENGTH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TOP_K_SUBREDDITS,
    MAX_HISTORY_LENGTH,
    METADATA_EXCLUDE_COLS,
    TEXT_COLS,
    TARGET_COL,
    ID_COL,
    RANDOM_STATE,
)
from library.utils import clean_text, set_seed


class FeatureGenerator:
    def __init__(self):
        set_seed(RANDOM_STATE)
        self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

        # Initialize VADER if possible
        self.sia = None
        if SentimentIntensityAnalyzer:
            try:
                # Check if lexicon exists, if not try to download
                try:
                    nltk.data.find("sentiment/vader_lexicon.zip")
                except LookupError:
                    nltk.download("vader_lexicon", quiet=True)
                self.sia = SentimentIntensityAnalyzer()
            except Exception:
                print(
                    "Warning: VADER lexicon not found and download failed. Sentiment features will be 0."
                )
                self.sia = None

    def _load_raw_data(self):
        """Loads and parses raw CSV data."""
        converters = {
            TEXT_COLS["subreddits"]: lambda x: (
                ast.literal_eval(x) if pd.notna(x) else []
            )
        }

        df_train = pd.read_csv(TRAIN_PATH, converters=converters)
        df_val = pd.read_csv(VAL_PATH, converters=converters)
        df_test = pd.read_csv(TEST_PATH, converters=converters)

        return df_train, df_val, df_test

    def _get_unique_subreddits(self, dfs):
        """Extracts all unique subreddits across datasets."""
        unique_subs = set()
        for df in dfs:
            for subs in df[TEXT_COLS["subreddits"]]:
                unique_subs.update(subs)
        return list(unique_subs)

    def _compute_embeddings(self, texts, batch_size=32):
        """Computes SBERT embeddings."""
        return self.sbert.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )

    def _compute_sentiment(self, text):
        """Computes VADER sentiment scores."""
        if self.sia is None or not text:
            return 0.0, 0.0
        scores = self.sia.polarity_scores(str(text))
        return scores["compound"], scores["neu"]

    def _process_history_consistency(self, df, sub_emb_map, title_embs, body_embs):
        """
        Computes Dispersion-Normalized Consistency metrics and prepares History Sequences.
        """
        n_samples = len(df)

        # Output arrays
        centroids = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)
        dispersions = np.zeros((n_samples, 1), dtype=np.float32)
        divergences = np.zeros((n_samples, 1), dtype=np.float32)
        z_scores = np.zeros((n_samples, 1), dtype=np.float32)
        history_seqs = np.zeros(
            (n_samples, MAX_HISTORY_LENGTH, EMBEDDING_DIM), dtype=np.float32
        )
        history_masks = np.zeros(
            (n_samples, MAX_HISTORY_LENGTH), dtype=np.float32
        )  # 1 for data, 0 for pad

        sub_col = TEXT_COLS["subreddits"]

        for i, row in df.iterrows():
            subs = row[sub_col]

            # Request Embedding (Avg of Title + Body)
            req_emb = (title_embs[i] + body_embs[i]) / 2.0

            if not subs:
                # No history: Neutral values
                divergences[i] = 0.0  # Or distance to origin? 0 seems safest neutral.
                z_scores[i] = 0.0
                # Centroid remains 0
                # Dispersion remains 0
                continue

            # Get history embeddings
            # Handle unknown subreddits (though we encoded all uniques, so should be fine)
            hist_embs = np.array([sub_emb_map[s] for s in subs if s in sub_emb_map])

            if len(hist_embs) == 0:
                continue

            # 1. Centroid
            centroid = np.mean(hist_embs, axis=0)
            centroids[i] = centroid

            # 2. Dispersion (Std dev of distances to centroid)
            dists = np.linalg.norm(hist_embs - centroid, axis=1)
            sigma = np.std(dists)
            dispersions[i] = sigma

            # 3. Request Divergence
            d = np.linalg.norm(req_emb - centroid)
            divergences[i] = d

            # 4. Z-Score
            # (Distance of request - Mean distance of history) / Dispersion
            # Note: Mean distance of history is NOT zero (it's mean of norms)
            mu_dist = np.mean(dists)
            z = (d - mu_dist) / (sigma + 1e-6)
            z_scores[i] = z

            # 5. History Sequence for MLP
            # Truncate or Pad
            seq_len = min(len(hist_embs), MAX_HISTORY_LENGTH)
            # Take last N items (assuming chronological, though order isn't guaranteed in JSON, usually implies recent)
            # We'll take the first N from the list provided.
            history_seqs[i, :seq_len, :] = hist_embs[:seq_len]
            history_masks[i, :seq_len] = 1.0

        return {
            "centroids": centroids,
            "dispersions": dispersions,
            "divergences": divergences,
            "z_scores": z_scores,
            "history_seqs": history_seqs,
            "history_masks": history_masks,
        }

    def _get_top_k_subreddits(self, df_train):
        """Identifies top K subreddits from training data."""
        all_subs = []
        for subs in df_train[TEXT_COLS["subreddits"]]:
            all_subs.extend(subs)

        counts = pd.Series(all_subs).value_counts()
        top_k = counts.head(TOP_K_SUBREDDITS).index.tolist()
        return top_k

    def _create_top_k_features(self, df, top_k_subs):
        """Creates binary indicator matrix for top K subreddits."""
        n = len(df)
        feats = np.zeros((n, len(top_k_subs)), dtype=np.float32)

        sub_to_idx = {s: i for i, s in enumerate(top_k_subs)}

        for i, subs in enumerate(df[TEXT_COLS["subreddits"]]):
            for s in subs:
                if s in sub_to_idx:
                    feats[i, sub_to_idx[s]] = 1.0
        return feats

    def _extract_metadata(self, df):
        """Extracts and engineers metadata features."""
        # 1. Select numerical columns (excluding ID, Target, etc.)
        num_cols = [
            c
            for c in df.columns
            if c not in METADATA_EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        meta_df = df[num_cols].copy()

        # 2. Text Stats
        title_col = TEXT_COLS["title"]
        body_col = TEXT_COLS["body"]

        # Clean text first
        titles = df[title_col].apply(clean_text)
        bodies = df[body_col].apply(clean_text)

        meta_df["title_len_char"] = titles.apply(len)
        meta_df["title_len_word"] = titles.apply(lambda x: len(x.split()))
        meta_df["body_len_char"] = bodies.apply(len)
        meta_df["body_len_word"] = bodies.apply(lambda x: len(x.split()))

        # Sentiment
        if self.sia:
            meta_df["title_sentiment"] = titles.apply(
                lambda x: self._compute_sentiment(x)[0]
            )
            meta_df["body_sentiment"] = bodies.apply(
                lambda x: self._compute_sentiment(x)[0]
            )
        else:
            meta_df["title_sentiment"] = 0.0
            meta_df["body_sentiment"] = 0.0

        return meta_df.values.astype(np.float32)

    def process(self, load_cached_data=True):
        """
        Main execution method.
        Checks cache, otherwise computes all features.
        """
        # Define cache paths
        cache_files = {
            "train": os.path.join(CACHE_DIR, "features_train.npz"),
            "val": os.path.join(CACHE_DIR, "features_val.npz"),
            "test": os.path.join(CACHE_DIR, "features_test.npz"),
        }

        # Check cache
        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            print("Loading features from cache...")
            data = {}
            for split, path in cache_files.items():
                data[split] = np.load(path, allow_pickle=True)
            return data["train"], data["val"], data["test"]

        print("Computing features from scratch...")

        # 1. Load Data
        df_train, df_val, df_test = self._load_raw_data()

        # 2. Prepare Text for SBERT
        print("Encoding text with SBERT...")
        # Titles
        train_titles = self._compute_embeddings(
            df_train[TEXT_COLS["title"]].apply(clean_text).tolist()
        )
        val_titles = self._compute_embeddings(
            df_val[TEXT_COLS["title"]].apply(clean_text).tolist()
        )
        test_titles = self._compute_embeddings(
            df_test[TEXT_COLS["title"]].apply(clean_text).tolist()
        )

        # Bodies
        train_bodies = self._compute_embeddings(
            df_train[TEXT_COLS["body"]].apply(clean_text).tolist()
        )
        val_bodies = self._compute_embeddings(
            df_val[TEXT_COLS["body"]].apply(clean_text).tolist()
        )
        test_bodies = self._compute_embeddings(
            df_test[TEXT_COLS["body"]].apply(clean_text).tolist()
        )

        # Subreddits
        print("Encoding subreddits...")
        all_dfs = [df_train, df_val, df_test]
        unique_subs = self._get_unique_subreddits(all_dfs)
        sub_embs = self._compute_embeddings(unique_subs)
        sub_emb_map = {s: emb for s, emb in zip(unique_subs, sub_embs)}

        # 3. History & Consistency Analysis
        print("Analyzing history consistency...")
        train_cons = self._process_history_consistency(
            df_train, sub_emb_map, train_titles, train_bodies
        )
        val_cons = self._process_history_consistency(
            df_val, sub_emb_map, val_titles, val_bodies
        )
        test_cons = self._process_history_consistency(
            df_test, sub_emb_map, test_titles, test_bodies
        )

        # 4. Top-K Subreddits
        print("Generating Top-K subreddit flags...")
        top_k = self._get_top_k_subreddits(df_train)
        train_topk = self._create_top_k_features(df_train, top_k)
        val_topk = self._create_top_k_features(df_val, top_k)
        test_topk = self._create_top_k_features(df_test, top_k)

        # 5. TF-IDF (RF Stream)
        print("Computing TF-IDF...")
        # Concat title and body
        train_text = (
            df_train[TEXT_COLS["title"]] + " " + df_train[TEXT_COLS["body"]]
        ).apply(clean_text)
        val_text = (df_val[TEXT_COLS["title"]] + " " + df_val[TEXT_COLS["body"]]).apply(
            clean_text
        )
        test_text = (
            df_test[TEXT_COLS["title"]] + " " + df_test[TEXT_COLS["body"]]
        ).apply(clean_text)

        self.tfidf.fit(train_text)
        train_tfidf = self.tfidf.transform(train_text)
        val_tfidf = self.tfidf.transform(val_text)
        test_tfidf = self.tfidf.transform(test_text)

        # 6. Metadata
        print("Extracting metadata...")
        train_meta_raw = self._extract_metadata(df_train)
        val_meta_raw = self._extract_metadata(df_val)
        test_meta_raw = self._extract_metadata(df_test)

        # Impute NaNs for RF
        self.imputer.fit(train_meta_raw)
        train_meta_imp = self.imputer.transform(train_meta_raw)
        val_meta_imp = self.imputer.transform(val_meta_raw)
        test_meta_imp = self.imputer.transform(test_meta_raw)

        # Scale for MLP (Arcsinh + StandardScaler)
        # Apply arcsinh to handle heavy tails before scaling
        train_meta_asc = np.arcsinh(train_meta_imp)
        val_meta_asc = np.arcsinh(val_meta_imp)
        test_meta_asc = np.arcsinh(test_meta_imp)

        self.scaler.fit(train_meta_asc)
        train_meta_scaled = self.scaler.transform(train_meta_asc)
        val_meta_scaled = self.scaler.transform(val_meta_asc)
        test_meta_scaled = self.scaler.transform(test_meta_asc)

        # 7. Assemble Features

        # --- Stream A: Random Forest ---
        # Stack: TFIDF (sparse) | TopK (dense) | Metadata (dense) | Consistency Metrics (dense)
        # Consistency metrics to include in RF: dispersion, z_score (divergence is raw distance, Z is better)

        def assemble_rf(tfidf, topk, meta, cons):
            # Extract consistency scalars
            cons_feats = np.hstack(
                [
                    cons["dispersions"],
                    cons["z_scores"],
                    cons["divergences"],  # Include raw divergence too
                ]
            )

            # Combine dense features
            dense = np.hstack([meta, topk, cons_feats])

            # Combine with sparse TFIDF
            return sparse.hstack([tfidf, dense]).tocsr()

        train_rf = assemble_rf(train_tfidf, train_topk, train_meta_imp, train_cons)
        val_rf = assemble_rf(val_tfidf, val_topk, val_meta_imp, val_cons)
        test_rf = assemble_rf(test_tfidf, test_topk, test_meta_imp, test_cons)

        # --- Stream B: MLP ---
        # Dictionary of arrays
        def assemble_mlp(titles, bodies, cons, meta_scaled, labels=None):
            data = {
                "title_emb": titles,
                "body_emb": bodies,
                "history_seq": cons["history_seqs"],
                "history_mask": cons["history_masks"],
                "centroid": cons["centroids"],
                "meta": meta_scaled,
            }
            if labels is not None:
                data["labels"] = labels
            return data

        train_labels = df_train[TARGET_COL].values.astype(np.float32)
        val_labels = df_val[TARGET_COL].values.astype(np.float32)

        train_mlp = assemble_mlp(
            train_titles, train_bodies, train_cons, train_meta_scaled, train_labels
        )
        val_mlp = assemble_mlp(
            val_titles, val_bodies, val_cons, val_meta_scaled, val_labels
        )
        test_mlp = assemble_mlp(
            test_titles, test_bodies, test_cons, test_meta_scaled
        )  # No labels for test

        # 8. Save to Cache
        print("Saving to cache...")

        def save_split(path, rf_data, mlp_data):
            # Save RF as sparse components or just rely on object saving (npz handles sparse if passed correctly? No.)
            # np.savez doesn't handle sparse matrices directly well.
            # We will save RF components separately or densify if small? No, TFIDF is 5000 dims.
            # Strategy: Save MLP dict items. Save RF as separate sparse file?
            # To keep it simple for npz: Save RF data, indices, indptr, shape.

            save_dict = {
                "rf_data": rf_data.data,
                "rf_indices": rf_data.indices,
                "rf_indptr": rf_data.indptr,
                "rf_shape": rf_data.shape,
                **mlp_data,
            }
            np.savez(path, **save_dict)

        save_split(cache_files["train"], train_rf, train_mlp)
        save_split(cache_files["val"], val_rf, val_mlp)
        save_split(cache_files["test"], test_rf, test_mlp)

        # Return loaded dicts (simulating load)
        return (
            np.load(cache_files["train"], allow_pickle=True),
            np.load(cache_files["val"], allow_pickle=True),
            np.load(cache_files["test"], allow_pickle=True),
        )


def get_features(load_cached_data=True):
    """Wrapper function to instantiate generator and get data."""
    generator = FeatureGenerator()
    return generator.process(load_cached_data=load_cached_data)
