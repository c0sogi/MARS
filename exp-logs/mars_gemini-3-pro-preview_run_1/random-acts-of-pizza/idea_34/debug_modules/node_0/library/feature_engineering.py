import os
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from library.config import Config
from library.utils import set_seed

# Ensure NLTK data is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class FeaturePipeline:
    """
    Feature engineering pipeline for the Hybrid Ensemble.
    Handles:
    1. SBERT Embedding (Title, Body, History)
    2. Predictive Top-K Subreddit Selection (Mutual Information)
    3. Peak-Relevance Scalar Calculation
    4. TF-IDF Vectorization
    5. Metadata & Sentiment Extraction
    """

    def __init__(self):
        self.top_k_subreddits = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.sbert_model = None
        self.vader = None
        self.mlb = None
        self.fitted = False

    def _init_models(self):
        """Lazy initialization of heavy models."""
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(Config.SBERT_MODEL_NAME)
        if self.vader is None:
            self.vader = SentimentIntensityAnalyzer()

    def fit_transform(self, df, split_name="train"):
        """
        Fits the pipeline on the training data and transforms it.
        Uses caching to avoid re-computation.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"features_{split_name}.npz")
        if os.path.exists(cache_path):
            print(f"Loading cached features for {split_name} from {cache_path}")
            # Even if we load cache, we must fit the internal state (vectorizers, etc.)
            # to ensure we can transform test data later.
            # However, re-fitting purely from the dataframe might be slow if we just want to load.
            # For this task, we assume if cache exists, we still need to fit the transformers
            # for the subsequent test set.
            # To save time, we will perform the fit logic but skip the heavy embedding/processing
            # if the cache is found, OR we assume the user runs train then test in one go.
            # We will proceed to fit the lightweight parts (TFIDF, MI) and then load the heavy parts.
            self._fit_estimators(df)
            return self._load_cache(cache_path)

        print(f"Generating features for {split_name}...")
        self._init_models()

        # Fit estimators (TF-IDF, MI Selector)
        self._fit_estimators(df)

        # Transform data
        data_dict = self._process_data(df, is_train=True)

        # Save to cache
        self._save_cache(cache_path, data_dict)
        self.fitted = True
        return data_dict

    def transform(self, df, split_name):
        """
        Transforms validation or test data using the fitted pipeline.
        """
        if not self.fitted and self.tfidf_vectorizer is None:
            raise RuntimeError(
                "Pipeline must be fitted on training data before transforming."
            )

        cache_path = os.path.join(Config.CACHE_DIR, f"features_{split_name}.npz")
        if os.path.exists(cache_path):
            print(f"Loading cached features for {split_name} from {cache_path}")
            return self._load_cache(cache_path)

        print(f"Generating features for {split_name}...")
        self._init_models()

        data_dict = self._process_data(df, is_train=False)

        self._save_cache(cache_path, data_dict)
        return data_dict

    def _fit_estimators(self, df):
        """Fits the TF-IDF and Subreddit Selectors."""
        print("Fitting estimators...")

        # 1. Predictive Top-K Subreddits (Mutual Information)
        # We only fit this if we have the target variable
        if Config.TARGET_COL in df.columns:
            print("  Fitting MI Subreddit Selector...")
            self.mlb = MultiLabelBinarizer(sparse_output=True)
            subs = df[Config.SUBREDDIT_COL].tolist()
            X_bin = self.mlb.fit_transform(subs)
            y = df[Config.TARGET_COL].astype(int).values

            # Calculate MI
            mi_scores = mutual_info_classif(
                X_bin, y, discrete_features=True, random_state=Config.RANDOM_STATE
            )

            # Select Top K
            # Handle case where we have fewer subreddits than K
            k = min(Config.TOP_K_MI_SUBREDDITS, len(self.mlb.classes_))
            top_k_indices = np.argsort(mi_scores)[-k:]
            self.top_k_subreddits = [self.mlb.classes_[i] for i in top_k_indices]
            print(f"  Selected {len(self.top_k_subreddits)} subreddits via MI.")
        else:
            print(
                "  Warning: Target column missing. Skipping MI fit (using existing or empty)."
            )
            if self.top_k_subreddits is None:
                self.top_k_subreddits = []

        # 2. TF-IDF
        print("  Fitting TF-IDF...")
        text_data = (
            df[Config.TEXT_COL_TITLE].fillna("")
            + " "
            + df[Config.TEXT_COL_BODY].fillna("")
        ).tolist()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_VOCAB_SIZE, stop_words="english"
        )
        self.tfidf_vectorizer.fit(text_data)

        # 3. Scaler initialization (fit later on processed metadata)
        if self.scaler is None:
            self.scaler = StandardScaler()

    def _process_data(self, df, is_train=False):
        """Core feature generation logic."""

        # --- A. Text Preparation ---
        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str).tolist()
        combined_text = [t + " " + b for t, b in zip(titles, bodies)]

        # --- B. SBERT Embeddings ---
        print("  Encoding text with SBERT...")
        title_emb = self.sbert_model.encode(
            titles, batch_size=64, show_progress_bar=False
        )
        body_emb = self.sbert_model.encode(
            bodies, batch_size=64, show_progress_bar=False
        )

        # --- C. History Embeddings & Peak Relevance ---
        print("  Processing user history...")
        # Optimization: Encode unique subreddits only
        all_subs = df[Config.SUBREDDIT_COL].explode().unique()
        all_subs = [s for s in all_subs if isinstance(s, str)]  # Filter non-strings

        if len(all_subs) > 0:
            sub_embeddings_map = {
                sub: emb
                for sub, emb in zip(
                    all_subs,
                    self.sbert_model.encode(
                        all_subs, batch_size=64, show_progress_bar=False
                    ),
                )
            }
        else:
            sub_embeddings_map = {}

        history_emb_list = []
        peak_sim_title = []
        peak_sim_body = []

        # Process per user
        for i, user_subs in enumerate(df[Config.SUBREDDIT_COL]):
            # Get valid embeddings for this user
            user_sub_embs = [
                sub_embeddings_map[s] for s in user_subs if s in sub_embeddings_map
            ]

            if len(user_sub_embs) > 0:
                user_sub_embs = np.array(user_sub_embs)  # (M, 384)
                history_emb_list.append(user_sub_embs)

                # Peak Relevance: Max Cosine Similarity
                # Title (1, 384) vs History (M, 384)
                sim_t = np.dot(user_sub_embs, title_emb[i])  # (M,) assuming normalized
                peak_sim_title.append(np.max(sim_t))

                sim_b = np.dot(user_sub_embs, body_emb[i])
                peak_sim_body.append(np.max(sim_b))
            else:
                # Handle empty history
                history_emb_list.append(np.zeros((0, 384), dtype=np.float32))
                peak_sim_title.append(0.0)
                peak_sim_body.append(0.0)

        peak_relevance = np.column_stack([peak_sim_title, peak_sim_body])

        # --- D. Metadata & Sentiment ---
        print("  Extracting metadata...")
        meta_features = []

        # Numerical columns
        num_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_upvotes_minus_downvotes_at_request",
        ]

        # Extract and Impute
        num_data = df[num_cols].copy()
        imputer = SimpleImputer(strategy="median")
        num_data = (
            imputer.fit_transform(num_data)
            if is_train
            else imputer.fit(df[num_cols]).transform(num_data)
        )  # Note: simple impute for now

        # Arcsinh Transform
        num_data = np.arcsinh(num_data)

        # Text Stats
        text_stats = []
        for t, b in zip(titles, bodies):
            text_stats.append(
                [
                    len(t),
                    len(b),
                    sum(1 for c in t if c.isupper()) / (len(t) + 1),
                    sum(1 for c in b if c.isupper()) / (len(b) + 1),
                ]
            )
        text_stats = np.array(text_stats)

        # Sentiment
        sent_scores = []
        for b in bodies:
            ss = self.vader.polarity_scores(b)
            sent_scores.append([ss["neg"], ss["neu"], ss["pos"], ss["compound"]])
        sent_scores = np.array(sent_scores)

        # Combine Metadata
        raw_metadata = np.hstack([num_data, text_stats, sent_scores])

        # Scale Metadata (Fit on train, transform on others)
        if is_train:
            scaled_metadata = self.scaler.fit_transform(raw_metadata)
        else:
            scaled_metadata = self.scaler.transform(raw_metadata)

        # --- E. TF-IDF & Top-K Flags (For RF) ---
        print("  Generating RF features...")
        tfidf_mat = self.tfidf_vectorizer.transform(combined_text).toarray()

        # Top-K Flags
        top_k_flags = []
        for user_subs in df[Config.SUBREDDIT_COL]:
            user_subs_set = set(user_subs)
            flags = [1 if sub in user_subs_set else 0 for sub in self.top_k_subreddits]
            top_k_flags.append(flags)
        top_k_flags = np.array(top_k_flags)

        # Assemble RF Features
        # [Metadata, PeakRelevance, TopK, TFIDF]
        rf_features = np.hstack(
            [scaled_metadata, peak_relevance, top_k_flags, tfidf_mat]
        ).astype(np.float32)

        # --- F. Assemble Output ---
        output = {
            "rf_features": rf_features,
            "mlp_metadata": scaled_metadata.astype(np.float32),
            "mlp_title_emb": title_emb.astype(np.float32),
            "mlp_body_emb": body_emb.astype(np.float32),
            "mlp_history_emb": np.array(history_emb_list, dtype=object),
            "peak_relevance": peak_relevance.astype(np.float32),
        }

        if Config.TARGET_COL in df.columns:
            output["labels"] = df[Config.TARGET_COL].astype(int).values

        return output

    def _save_cache(self, path, data_dict):
        """Saves data dictionary to compressed npz."""
        np.savez_compressed(path, **data_dict)
        print(f"Saved features to {path}")

    def _load_cache(self, path):
        """Loads data dictionary from npz."""
        loaded = np.load(path, allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}
