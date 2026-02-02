import os
import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library.config import PathConfig, FeatureConfig, TrainingConfig
from library.utils import save_numpy, load_numpy, seed_everything


class TextFeatureExtractor:
    """
    Handles text processing for both Lexical (Stream A) and Semantic (Stream B) pipelines.
    """

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=FeatureConfig.TFIDF_MAX_FEATURES,
            ngram_range=FeatureConfig.TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )
        self.sbert_model = None  # Lazy load

    def _get_combined_text(self, df):
        """Concatenates title and edit-aware text."""
        # Fill NaNs with empty string
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return title + " " + body

    def extract_meta_features(self, df):
        """
        Extracts manual features: char length, word count, caps ratio, etc.
        Returns a numpy array of shape (N, 3).
        """
        text = self._get_combined_text(df)

        # 1. Character Length
        char_len = text.apply(len).values.reshape(-1, 1)

        # 2. Word Count
        word_count = text.apply(lambda x: len(x.split())).values.reshape(-1, 1)

        # 3. Caps Ratio (proxy for urgency/shouting)
        def get_caps_ratio(s):
            if len(s) == 0:
                return 0.0
            return sum(1 for c in s if c.isupper()) / len(s)

        caps_ratio = text.apply(get_caps_ratio).values.reshape(-1, 1)

        return np.hstack([char_len, word_count, caps_ratio])

    def fit_transform_tfidf(self, train_df, val_df, test_df):
        """Generates TF-IDF features for Stream A."""
        train_text = self._get_combined_text(train_df)
        val_text = self._get_combined_text(val_df)
        test_text = self._get_combined_text(test_df)

        print("Fitting TF-IDF Vectorizer...")
        X_train = (
            self.tfidf_vectorizer.fit_transform(train_text).toarray().astype(np.float32)
        )
        X_val = self.tfidf_vectorizer.transform(val_text).toarray().astype(np.float32)
        X_test = self.tfidf_vectorizer.transform(test_text).toarray().astype(np.float32)

        return X_train, X_val, X_test

    def generate_sbert_embeddings(self, train_df, val_df, test_df):
        """Generates SBERT embeddings for Stream B."""
        if self.sbert_model is None:
            print(f"Loading SBERT model: {FeatureConfig.SBERT_MODEL_NAME}...")
            self.sbert_model = SentenceTransformer(FeatureConfig.SBERT_MODEL_NAME)
            if torch.cuda.is_available():
                self.sbert_model = self.sbert_model.to("cuda")

        train_text = self._get_combined_text(train_df).tolist()
        val_text = self._get_combined_text(val_df).tolist()
        test_text = self._get_combined_text(test_df).tolist()

        print("Encoding text with SBERT (this may take a while)...")
        # Batch size can be adjusted based on GPU VRAM
        X_train = self.sbert_model.encode(
            train_text, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        X_val = self.sbert_model.encode(
            val_text, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        X_test = self.sbert_model.encode(
            test_text, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

        return X_train, X_val, X_test


class SubredditEncoder:
    """
    Handles community feature extraction: Multi-Hot (Stream A) and Integer Sequence (Stream B).
    """

    def __init__(self):
        self.vocab = {}
        self.id_map = {}  # sub -> int
        self.top_k = FeatureConfig.TOP_K_SUBREDDITS
        self.mlb = None

    def _parse_subreddits(self, df):
        """Parses the stringified list of subreddits."""
        raw_col = df[FeatureConfig.SUBREDDIT_COL]
        # Handle cases where it might already be a list or needs eval
        parsed = []
        for x in raw_col:
            if isinstance(x, str):
                try:
                    # Safe eval of string representation of list
                    res = ast.literal_eval(x)
                    if not isinstance(res, list):
                        res = []
                except:
                    res = []
            elif isinstance(x, list):
                res = x
            else:
                res = []
            parsed.append([str(s).lower() for s in res])  # Normalize to lower case
        return parsed

    def fit(self, train_df):
        """Builds vocabulary from training data."""
        print("Building Subreddit Vocabulary...")
        subs_list = self._parse_subreddits(train_df)

        # Count frequencies
        counts = {}
        for user_subs in subs_list:
            for sub in user_subs:
                counts[sub] = counts.get(sub, 0) + 1

        # Select Top K
        sorted_subs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_subs = [s for s, c in sorted_subs[: self.top_k]]

        # 0 is PAD, 1 is UNK, 2...K+1 are top subs
        self.id_map = {sub: i + 2 for i, sub in enumerate(top_subs)}
        self.vocab_list = top_subs  # For MultiLabelBinarizer

        # Fit MultiLabelBinarizer for Stream A
        self.mlb = MultiLabelBinarizer(classes=top_subs, sparse_output=False)
        # MLB fit takes a list of iterables. It only cares about classes provided in init.
        # We don't strictly need to call fit if we provided classes, but let's be safe.
        self.mlb.fit([[]])

    def transform_multihot(self, df):
        """Stream A: Multi-Hot Encoding."""
        subs_list = self._parse_subreddits(df)
        # Filter subs to only those in top_k for MLB to work cleanly without warnings
        # actually MLB ignores unknown classes if we set classes in init?
        # No, MLB with fixed classes ignores others.

        # However, to be precise, let's just pass the list.
        # MLB will produce a binary matrix for the classes we defined.
        return self.mlb.transform(subs_list).astype(np.float32)

    def transform_sequences(self, df, max_len=50):
        """Stream B: Integer Sequences with Padding."""
        subs_list = self._parse_subreddits(df)

        N = len(df)
        X_seq = np.zeros((N, max_len), dtype=np.int64)  # 0 is PAD

        for i, user_subs in enumerate(subs_list):
            seq = []
            for sub in user_subs:
                idx = self.id_map.get(sub, 1)  # 1 is UNK
                seq.append(idx)

            # Truncate or Pad
            if len(seq) > max_len:
                seq = seq[:max_len]

            X_seq[i, : len(seq)] = seq

        return X_seq


class NumericalPreprocessor:
    """
    Handles numerical features: Imputation and Scaling.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cols = FeatureConfig.NUMERIC_COLS

    def fit_transform(self, train_df, val_df, test_df):
        # Extract raw
        X_train_raw = train_df[self.cols].values
        X_val_raw = val_df[self.cols].values
        X_test_raw = test_df[self.cols].values

        # Impute (Shared for both streams)
        print("Imputing numerical features...")
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        X_val_imp = self.imputer.transform(X_val_raw)
        X_test_imp = self.imputer.transform(X_test_raw)

        # Scale (Only for Stream B / MLP)
        print("Scaling numerical features...")
        X_train_sc = self.scaler.fit_transform(X_train_imp)
        X_val_sc = self.scaler.transform(X_val_imp)
        X_test_sc = self.scaler.transform(X_test_imp)

        return (X_train_imp, X_val_imp, X_test_imp), (X_train_sc, X_val_sc, X_test_sc)


def run_feature_engineering(load_cached_data: bool = True):
    """
    Main orchestration function.
    Checks cache, loads data, runs extractors, saves cache.
    Returns dictionaries for Stream A and Stream B data.
    """
    seed_everything(TrainingConfig.SEED)

    # Define Cache Paths
    # Stream A: Lexical/Sparse (RF)
    path_a_train = PathConfig.STREAM_A_PREFIX + "train.npz"
    path_a_val = PathConfig.STREAM_A_PREFIX + "val.npz"
    path_a_test = PathConfig.STREAM_A_PREFIX + "test.npz"

    # Stream B: Semantic/Dense (MLP)
    path_b_train = PathConfig.STREAM_B_PREFIX + "train.npz"
    path_b_val = PathConfig.STREAM_B_PREFIX + "val.npz"
    path_b_test = PathConfig.STREAM_B_PREFIX + "test.npz"

    all_paths = [
        path_a_train,
        path_a_val,
        path_a_test,
        path_b_train,
        path_b_val,
        path_b_test,
    ]
    cache_exists = all(os.path.exists(p) for p in all_paths)

    if load_cached_data and cache_exists:
        print("Loading features from cache...")
        return (
            load_numpy(path_a_train),
            load_numpy(path_a_val),
            load_numpy(path_a_test),
            load_numpy(path_b_train),
            load_numpy(path_b_val),
            load_numpy(path_b_test),
        )

    print("Cache missing or reload forced. Starting Feature Engineering...")

    # Load Metadata
    print("Loading Metadata CSVs...")
    df_train = pd.read_csv(PathConfig.TRAIN_CSV)
    df_val = pd.read_csv(PathConfig.VAL_CSV)
    df_test = pd.read_csv(PathConfig.TEST_CSV)

    # --- 1. Text Processing ---
    print("\n--- Processing Text Features ---")
    text_extractor = TextFeatureExtractor()

    # Meta Features (Shared logic, but computed per split)
    meta_train = text_extractor.extract_meta_features(df_train)
    meta_val = text_extractor.extract_meta_features(df_val)
    meta_test = text_extractor.extract_meta_features(df_test)

    # Stream A: TF-IDF
    tfidf_train, tfidf_val, tfidf_test = text_extractor.fit_transform_tfidf(
        df_train, df_val, df_test
    )

    # Stream B: SBERT
    sbert_train, sbert_val, sbert_test = text_extractor.generate_sbert_embeddings(
        df_train, df_val, df_test
    )

    # --- 2. Community Processing ---
    print("\n--- Processing Community Features ---")
    sub_encoder = SubredditEncoder()
    sub_encoder.fit(df_train)

    # Stream A: Multi-Hot
    comm_mh_train = sub_encoder.transform_multihot(df_train)
    comm_mh_val = sub_encoder.transform_multihot(df_val)
    comm_mh_test = sub_encoder.transform_multihot(df_test)

    # Stream B: Sequences
    comm_seq_train = sub_encoder.transform_sequences(df_train)
    comm_seq_val = sub_encoder.transform_sequences(df_val)
    comm_seq_test = sub_encoder.transform_sequences(df_test)

    # --- 3. Numerical Processing ---
    print("\n--- Processing Numerical Features ---")
    num_proc = NumericalPreprocessor()
    (num_imp_train, num_imp_val, num_imp_test), (
        num_sc_train,
        num_sc_val,
        num_sc_test,
    ) = num_proc.fit_transform(df_train, df_val, df_test)

    # --- 4. Targets & IDs ---
    y_train = df_train[FeatureConfig.TARGET_COL].astype(int).values
    y_val = df_val[FeatureConfig.TARGET_COL].astype(int).values
    # Test has no target usually, but we might need IDs
    ids_test = df_test[FeatureConfig.ID_COL].values

    # --- 5. Packaging & Saving ---
    print("\nSaving processed features...")

    # Stream A Data (RF)
    # Concatenate Meta + Numerical(Imputed) for RF 'meta' input
    # RF inputs: TFIDF (sparse/dense), Community (dense), Meta+Num (dense)
    # We save them separately to allow flexibility in the model loader
    data_a_train = {
        "tfidf": tfidf_train,
        "community": comm_mh_train,
        "meta_num": np.hstack([meta_train, num_imp_train]),
        "y": y_train,
    }
    data_a_val = {
        "tfidf": tfidf_val,
        "community": comm_mh_val,
        "meta_num": np.hstack([meta_val, num_imp_val]),
        "y": y_val,
    }
    data_a_test = {
        "tfidf": tfidf_test,
        "community": comm_mh_test,
        "meta_num": np.hstack([meta_test, num_imp_test]),
        "ids": ids_test,
    }

    # Stream B Data (MLP)
    data_b_train = {
        "sbert": sbert_train,
        "community": comm_seq_train,
        "meta_num": np.hstack([meta_train, num_sc_train]),  # Scaled for MLP
        "y": y_train,
    }
    data_b_val = {
        "sbert": sbert_val,
        "community": comm_seq_val,
        "meta_num": np.hstack([meta_val, num_sc_val]),
        "y": y_val,
    }
    data_b_test = {
        "sbert": sbert_test,
        "community": comm_seq_test,
        "meta_num": np.hstack([meta_test, num_sc_test]),
        "ids": ids_test,
    }

    save_numpy(data_a_train, path_a_train)
    save_numpy(data_a_val, path_a_val)
    save_numpy(data_a_test, path_a_test)

    save_numpy(data_b_train, path_b_train)
    save_numpy(data_b_val, path_b_val)
    save_numpy(data_b_test, path_b_test)

    print("Feature Engineering Complete.")

    # Reload to ensure consistency with return signature
    return (
        load_numpy(path_a_train),
        load_numpy(path_a_val),
        load_numpy(path_a_test),
        load_numpy(path_b_train),
        load_numpy(path_b_val),
        load_numpy(path_b_test),
    )
