import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import seed_everything


class FeaturePipeline:
    """
    Handles data loading, text vectorization using Sentence Transformers,
    concatenation, scaling, and caching of features.
    """

    def __init__(self):
        seed_everything(Config.SEED)

    def process_data(
        self, load_cached_data: bool = True, debug_sample_size: int = None
    ):
        """
        Main method to get processed data.

        Args:
            load_cached_data (bool): Whether to try loading from cache.
            debug_sample_size (int, optional): If set, only process a subset of data for debugging.
                                               Disables loading from/saving to main cache.

        Returns:
            Tuple containing:
            - X_train (np.ndarray): Scaled training features.
            - y_train (np.ndarray): Training targets.
            - X_val (np.ndarray): Scaled validation features.
            - y_val (np.ndarray): Validation targets.
            - X_test (np.ndarray): Scaled test features.
            - test_ids (np.ndarray): Test IDs.
        """
        # If debugging, we don't use the main cache to avoid corruption
        if debug_sample_size is not None:
            return self._compute_features(
                debug_sample_size=debug_sample_size, save_to_cache=False
            )

        # Check cache
        if load_cached_data and self._check_cache_exists():
            print("Loading cached features from disk...")
            return self._load_cache()

        # Compute and cache
        print("Cache not found or forced reload. Computing features...")
        return self._compute_features(save_to_cache=True)

    def _check_cache_exists(self) -> bool:
        """Checks if all required cache files exist."""
        required_files = [
            Config.TRAIN_EMBEDS_PATH,
            Config.TRAIN_LABELS_PATH,
            Config.VAL_EMBEDS_PATH,
            Config.VAL_LABELS_PATH,
            Config.TEST_EMBEDS_PATH,
            Config.TEST_IDS_PATH,
        ]
        return all(os.path.exists(f) for f in required_files)

    def _load_cache(self):
        """Loads features from .npy files."""
        X_train = np.load(Config.TRAIN_EMBEDS_PATH)
        y_train = np.load(Config.TRAIN_LABELS_PATH)
        X_val = np.load(Config.VAL_EMBEDS_PATH)
        y_val = np.load(Config.VAL_LABELS_PATH)
        X_test = np.load(Config.TEST_EMBEDS_PATH)
        test_ids = np.load(Config.TEST_IDS_PATH)
        return X_train, y_train, X_val, y_val, X_test, test_ids

    def _compute_features(
        self, debug_sample_size: int = None, save_to_cache: bool = True
    ):
        """
        Computes embeddings, concatenates, scales, and optionally saves to cache.
        """
        # 1. Load Data
        train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
        val_df = pd.read_csv(Config.VAL_DATA_PATH)
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # Debug sampling
        if debug_sample_size is not None:
            train_df = train_df.head(debug_sample_size)
            val_df = val_df.head(debug_sample_size)
            test_df = test_df.head(debug_sample_size)

        # 2. Initialize Model
        print(f"Loading SentenceTransformer: {Config.SENTENCE_TRANSFORMER_MODEL}")
        model = SentenceTransformer(
            Config.SENTENCE_TRANSFORMER_MODEL, device=Config.DEVICE
        )

        # 3. Vectorization (Encode -> Concat)
        print("Encoding Training Data...")
        X_train = self._vectorize_text(train_df, model)
        print("Encoding Validation Data...")
        X_val = self._vectorize_text(val_df, model)
        print("Encoding Test Data...")
        X_test = self._vectorize_text(test_df, model)

        # 4. Extract Targets/IDs
        target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
        y_train = train_df[target_cols].values.astype(np.float32)
        y_val = val_df[target_cols].values.astype(np.float32)
        test_ids = test_df["id"].values

        # 5. Scaling (Fit on Train, Transform All)
        print("Normalizing features (StandardScaler)...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # 6. Cache Results
        if save_to_cache:
            print(f"Saving features to {Config.CACHE_DIR}...")
            # Ensure directory exists
            os.makedirs(Config.CACHE_DIR, exist_ok=True)

            np.save(Config.TRAIN_EMBEDS_PATH, X_train)
            np.save(Config.TRAIN_LABELS_PATH, y_train)
            np.save(Config.VAL_EMBEDS_PATH, X_val)
            np.save(Config.VAL_LABELS_PATH, y_val)
            np.save(Config.TEST_EMBEDS_PATH, X_test)
            np.save(Config.TEST_IDS_PATH, test_ids)

        return X_train, y_train, X_val, y_val, X_test, test_ids

    def _vectorize_text(
        self, df: pd.DataFrame, model: SentenceTransformer
    ) -> np.ndarray:
        """
        Encodes prompt, response_a, and response_b, then concatenates them.
        Returns a numpy array of shape (N, embedding_dim * 3).
        """
        # Ensure text columns are strings and handle potential NaNs
        prompts = df["prompt"].fillna("").astype(str).tolist()
        res_a = df["response_a"].fillna("").astype(str).tolist()
        res_b = df["response_b"].fillna("").astype(str).tolist()

        # Encode
        # show_progress_bar=False to reduce clutter as requested
        emb_p = model.encode(
            prompts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emb_a = model.encode(
            res_a,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emb_b = model.encode(
            res_b,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 1. Compute Orthogonal Structural Features (Cite solution_lesson_node_00006)
        # We replace derivative cosine similarities with structural metadata like newline counts.

        def get_len_features(text_list):
            # Character count
            chars = np.array([len(t) for t in text_list]).reshape(-1, 1)
            # Word count (simple whitespace split)
            words = np.array([len(t.split()) for t in text_list]).reshape(-1, 1)
            # Newline count (proxy for formatting/code)
            newlines = np.array([t.count("\n") for t in text_list]).reshape(-1, 1)
            return chars, words, newlines

        len_char_a, len_word_a, newline_a = get_len_features(res_a)
        len_char_b, len_word_b, newline_b = get_len_features(res_b)

        # Differences (A - B)
        diff_len_char = len_char_a - len_char_b
        diff_len_word = len_word_a - len_word_b

        # Length Ratio (Cite solution_lesson_node_00006)
        # Adding 1.0 to denominator to avoid division by zero
        len_ratio = len_char_a / (len_char_b + 1.0)

        # 2. Concatenate: [Embeddings, Lengths, Diffs, Newlines, Ratio]
        # Total extra features: 4 (raw lens) + 2 (diffs) + 2 (newlines) + 1 (ratio) = 9
        extra_features = np.hstack(
            [
                len_char_a,
                len_char_b,
                len_word_a,
                len_word_b,
                diff_len_char,
                diff_len_word,
                newline_a,
                newline_b,
                len_ratio,
            ]
        )

        features = np.hstack([emb_p, emb_a, emb_b, extra_features])
        return features.astype(np.float32)
