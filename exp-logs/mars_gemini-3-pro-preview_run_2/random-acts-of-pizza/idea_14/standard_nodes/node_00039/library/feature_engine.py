import os
import string
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import (
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TRANSFORMER_MODEL,
    SEMANTIC_TEXT_COLS,
    STRUCTURAL_COLS,
    TEMPORAL_COLS,
    TEMPORAL_RAW_COL,
    USER_METADATA_COLS,
    SEED,
)
from library.utils import load_data, save_data, set_seed
from library.data_loader import get_dataset, load_split_metadata

# Set global seed for reproducibility
set_seed(SEED)


class TextEmbedder:
    """
    Handles generation of semantic embeddings using a pre-trained Transformer.
    Includes L2 normalization as per the FABLE strategy.
    """

    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(TRANSFORMER_MODEL)
        return cls._model

    def transform(self, texts):
        """
        Generates L2-normalized embeddings for a list/series of texts.

        Args:
            texts (list or pd.Series): Input texts.

        Returns:
            np.ndarray: L2-normalized embeddings matrix.
        """
        model = self._get_model()
        # Encode texts
        embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        # Apply L2 Normalization (project to hypersphere)
        return normalize(embeddings, norm="l2")


def extract_temporal_features(df):
    """
    Extracts View 3: Temporal Context.
    Only uses raw timestamp as per Lesson 00033.
    """
    return df[[TEMPORAL_RAW_COL]].copy()


def extract_user_metadata(df):
    """
    Extracts View 4: User Metadata.
    Selects existing numerical columns.
    """
    # Select columns, fill NaNs with 0 (though analysis showed none, safety first)
    features = df[USER_METADATA_COLS].copy()
    features = features.fillna(0)
    return features


class ScalerWrapper:
    """
    Wraps QuantileTransformer (RankGauss) to scale specific feature views.
    Ensures that the scaler is fit only on training data and applied consistently.
    """

    def __init__(self, columns_to_scale):
        self.columns = columns_to_scale
        # Output distribution 'normal' implements RankGauss
        self.scaler = QuantileTransformer(
            output_distribution="normal",
            random_state=SEED,
            n_quantiles=1000,  # Default, will auto-adjust if n_samples < 1000
        )

    def fit(self, df):
        """Fits the scaler on the specified columns of the dataframe."""
        if not self.columns:
            return self

        X = df[self.columns]
        self.scaler.fit(X)
        return self

    def transform(self, df):
        """Transforms the specified columns of the dataframe."""
        if not self.columns:
            return df

        df_transformed = df.copy()
        df_transformed[self.columns] = self.scaler.transform(df[self.columns])
        return df_transformed


def generate_features(split, load_cached_data=True, debug=False, debug_size=100):
    """
    Main pipeline function to generate or load features for a given split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.
        debug (bool): If True, runs on a subset and does not save to main cache.
        debug_size (int): Number of samples for debug mode.

    Returns:
        pd.DataFrame: DataFrame containing request_id, target (if available), and all feature views.
    """
    # Determine cache path based on split
    if split == "train":
        cache_path = TRAIN_FEATURES_PATH
    elif split == "val":
        cache_path = VAL_FEATURES_PATH
    elif split == "test":
        cache_path = TEST_FEATURES_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # If debugging, we ignore the main cache to avoid pollution and force re-computation
    if debug:
        load_cached_data = False

    # Try loading from cache with validation (Cite solution_lesson_node_00034)
    if load_cached_data and os.path.exists(cache_path):
        if debug:
            # In debug mode, we don't validate against full metadata length
            print(
                f"[{split.upper()}] Loading features from cache (Debug Mode): {cache_path}"
            )
            return load_data(cache_path)
        else:
            # Validate cache integrity
            df_meta = load_split_metadata(split)
            expected_rows = len(df_meta)
            df_cached = load_data(cache_path)

            if len(df_cached) == expected_rows:
                print(f"[{split.upper()}] Loading features from cache: {cache_path}")
                return df_cached
            else:
                print(
                    f"[{split.upper()}] Cache mismatch ({len(df_cached)} vs {expected_rows}). Regenerating..."
                )

    print(f"[{split.upper()}] Generating features from scratch (Debug={debug})...")

    # 1. Load Raw Data
    df_raw = get_dataset(split, debug=debug, debug_size=debug_size)

    # 2. Generate View 1: Semantic Embeddings
    # Concatenate Title + Text
    print(f"[{split.upper()}] Generating Semantic Embeddings (View 1)...")
    text_input = (
        df_raw[SEMANTIC_TEXT_COLS[0]].fillna("")
        + " "
        + df_raw[SEMANTIC_TEXT_COLS[1]].fillna("")
    ).tolist()

    embedder = TextEmbedder()
    embeddings = embedder.transform(text_input)

    # Convert embeddings to DataFrame with named columns
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df_raw.index)

    # 3. Generate View 2: Structural Features - SKIPPED (Cite solution_lesson_node_00038)

    # 4. Generate View 3: Temporal Features
    print(f"[{split.upper()}] Generating Temporal Features (View 3)...")
    df_temp = extract_temporal_features(df_raw)

    # 5. Generate View 4: User Metadata
    print(f"[{split.upper()}] Generating User Metadata (View 4)...")
    df_meta = extract_user_metadata(df_raw)

    # 6. Assemble Final DataFrame
    # Start with identifiers and target
    meta_cols = ["request_id"]
    if "requester_received_pizza" in df_raw.columns:
        meta_cols.append("requester_received_pizza")

    df_final = pd.concat([df_raw[meta_cols], df_emb, df_temp, df_meta], axis=1)

    # 7. Cache Result (Only if not debugging)
    if not debug:
        print(f"[{split.upper()}] Saving features to cache: {cache_path}")
        save_data(df_final, cache_path)

    return df_final
