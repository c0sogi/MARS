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

        # 3.1 Extract Scalar Features (Cite solution_lesson_node_00002)
        # Adding explicit interaction features like length differences
        print("Extracting Scalar Features...")
        S_train = self._extract_scalars(train_df)
        S_val = self._extract_scalars(val_df)
        S_test = self._extract_scalars(test_df)

        # Concatenate Embeddings and Scalars
        X_train = np.hstack([X_train, S_train])
        X_val = np.hstack([X_val, S_val])
        X_test = np.hstack([X_test, S_test])

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

        # Concatenate: [Prompt, Response_A, Response_B]
        features = np.hstack([emb_p, emb_a, emb_b])
        return features.astype(np.float32)

    def _extract_scalars(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts scalar features such as lengths and differences.
        Cite solution_lesson_node_00002: Interaction features like length differences
        are crucial for preference modeling.
        """
        # Fill NaNs
        df_clean = df.fillna("")

        # Lengths
        len_p = df_clean["prompt"].astype(str).apply(len).values
        len_a = df_clean["response_a"].astype(str).apply(len).values
        len_b = df_clean["response_b"].astype(str).apply(len).values

        # Newline counts (structure proxy)
        nl_a = df_clean["response_a"].astype(str).apply(lambda x: x.count("\n")).values
        nl_b = df_clean["response_b"].astype(str).apply(lambda x: x.count("\n")).values

        # Derived features
        len_diff = len_a - len_b
        len_ratio = len_a / (len_b + 1.0)  # Avoid div by zero

        # Stack features: [len_p, len_a, len_b, len_diff, len_ratio, nl_a, nl_b]
        # Shape: (N, 7)
        features = np.column_stack(
            [len_p, len_a, len_b, len_diff, len_ratio, nl_a, nl_b]
        )

        return features.astype(np.float32)
