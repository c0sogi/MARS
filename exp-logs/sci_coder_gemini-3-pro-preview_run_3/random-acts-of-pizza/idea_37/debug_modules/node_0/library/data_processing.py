import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library import config
from library import utils


class FeatureExtractor:
    def __init__(self):
        # 1. Sparse Text Branch
        self.text_vectorizer = TfidfVectorizer(**config.TEXT_TFIDF_PARAMS)

        # 2. Sparse Behavioral Branch
        self.community_vectorizer = TfidfVectorizer(**config.COMMUNITY_TFIDF_PARAMS)

        # 3. Dense Text Branch
        self.embedding_model = None  # Lazy load
        self.semantic_scaler = StandardScaler()

        # 4. Metadata Branch
        self.meta_imputer = SimpleImputer(strategy="median")
        self.meta_scaler = StandardScaler()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_embedding_model(self):
        if self.embedding_model is None:
            print(
                f"[FeatureExtractor] Loading embedding model {config.EMBEDDING_MODEL} on {self.device}..."
            )
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL, device=self.device
            )

    def _prepare_text(self, df):
        # Concatenate title and body
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return (title + " " + body).tolist()

    def _prepare_community(self, df):
        # Convert list of subreddits to space-separated string for TF-IDF
        # Handle cases where column might be NaN or empty lists
        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            return ""

        return df["requester_subreddits_at_request"].apply(join_subs).tolist()

    def _prepare_metadata(self, df):
        # Extract allow-listed columns
        meta_df = df[config.METADATA_COLS].copy()
        # Convert to float for safety
        return meta_df.astype(float).values

    def fit(self, df):
        print("[FeatureExtractor] Fitting transformers...")

        # 1. Metadata
        meta_data = self._prepare_metadata(df)
        self.meta_imputer.fit(meta_data)
        meta_imputed = self.meta_imputer.transform(meta_data)
        self.meta_scaler.fit(meta_imputed)

        # 2. Sparse Text
        text_data = self._prepare_text(df)
        self.text_vectorizer.fit(text_data)

        # 3. Sparse Community
        community_data = self._prepare_community(df)
        self.community_vectorizer.fit(community_data)

        # 4. Dense Text (Embeddings)
        # We need to compute embeddings on train to fit the scaler
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            text_data, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        self.semantic_scaler.fit(embeddings)

        print("[FeatureExtractor] Fitting complete.")
        return self

    def transform(self, df):
        # 1. Metadata
        meta_data = self._prepare_metadata(df)
        meta_imputed = self.meta_imputer.transform(meta_data)
        X_meta = self.meta_scaler.transform(meta_imputed)

        # 2. Sparse Text
        text_data = self._prepare_text(df)
        X_lexical = self.text_vectorizer.transform(text_data)

        # 3. Sparse Community
        community_data = self._prepare_community(df)
        X_community = self.community_vectorizer.transform(community_data)

        # 4. Dense Text
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            text_data, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        X_semantic = self.semantic_scaler.transform(embeddings)

        # Output dict
        output = {
            "metadata": X_meta.astype(np.float32),
            "lexical": X_lexical,  # Keep sparse
            "community": X_community,  # Keep sparse
            "semantic": X_semantic.astype(np.float32),
        }

        # Add target if present
        if config.TARGET_COL in df.columns:
            output["y"] = df[config.TARGET_COL].values.astype(int)

        # Add IDs for tracking
        if config.ID_COL in df.columns:
            output["ids"] = df[config.ID_COL].values

        return output


def get_processed_data(load_cached_data=True):
    """
    Orchestrates data loading, processing, and caching.
    Returns dictionaries for train, val, and test splits.
    """
    splits = ["train", "val", "test"]
    data_objects = {}

    # Define filenames for cache
    # We store each view separately to handle sparse formats correctly in utils.save_cache
    keys = ["metadata", "lexical", "community", "semantic", "y", "ids"]

    # Check if cache exists for all splits
    cache_complete = True
    if load_cached_data:
        for split in splits:
            for key in keys:
                # y is not in test
                if split == "test" and key == "y":
                    continue

                # Determine extension based on expected type
                ext = ".npz" if key in ["lexical", "community"] else ".npy"
                filename = f"X_{split}_{key}{ext}"

                if not os.path.exists(os.path.join(config.CACHE_DIR, filename)):
                    cache_complete = False
                    break
            if not cache_complete:
                break
    else:
        cache_complete = False

    if load_cached_data and cache_complete:
        print("[DataProcessing] Loading cached data...")
        for split in splits:
            split_data = {}
            for key in keys:
                if split == "test" and key == "y":
                    continue

                ext = ".npz" if key in ["lexical", "community"] else ".npy"
                filename = f"X_{split}_{key}{ext}"

                # Load using utils
                loaded = utils.load_cache(filename)

                # If loading sparse matrix from npz (saved via savez_compressed),
                # utils.load_cache returns NpzFile. We need to reconstruct if it was sparse,
                # but here utils.load_cache returns np.load result.
                # For sparse matrices saved as .npz (scipy style), usually we use scipy.sparse.load_npz.
                # However, utils.save_cache uses np.savez_compressed for dicts.
                # The FeatureExtractor returns scipy sparse matrices for lexical/community.
                # To be safe with the provided utils.save_cache which expects dict for .npz:
                # We will handle the saving/loading logic below carefully.
                # Wait, utils.save_cache: "if isinstance(data, dict): np.savez_compressed".
                # Scipy sparse matrices are not dicts.
                # We should convert sparse matrices to a dict representation or densify if small (but they aren't).
                # Actually, best practice with provided utils is to let the transformation logic handle it.
                # But since I cannot change utils, I must adapt.
                # Scipy sparse matrices can be saved as npz using scipy.sparse.save_npz,
                # but utils.save_cache uses np.savez_compressed.
                # I will wrap sparse matrices in a dict for saving: {'data': data, 'indices': indices, ...}
                # OR simpler: Since I implement get_processed_data, I can control what I pass to save_cache.

                # Let's assume for this implementation that we re-compute if cache logic is too complex
                # given the rigid utils, BUT the prompt asks to strictly follow logic.
                # I will assume utils.load_cache returns the object.
                # If it's a dict (from npz), I might need to reconstruct.
                # Let's rely on the fact that I will save them as dicts of arrays if needed.

                split_data[key] = loaded

                # Reconstruct sparse if it was saved as components
                if key in ["lexical", "community"] and isinstance(
                    loaded, np.lib.npyio.NpzFile
                ):
                    # Reconstruct scipy sparse matrix
                    from scipy import sparse

                    try:
                        split_data[key] = sparse.csr_matrix(
                            (loaded["data"], loaded["indices"], loaded["indptr"]),
                            shape=loaded["shape"],
                        )
                    except KeyError:
                        # Fallback if not saved as sparse components
                        pass

            data_objects[split] = split_data

        print("[DataProcessing] Cache loaded successfully.")
        return data_objects["train"], data_objects["val"], data_objects["test"]

    # --- Compute from scratch ---
    print("[DataProcessing] Cache miss or reload forced. Processing from scratch...")

    # 1. Load Raw Data
    df_train = utils.load_dataset("train")
    df_val = utils.load_dataset("val")
    df_test = utils.load_dataset("test")

    # 2. Fit Feature Extractor
    extractor = FeatureExtractor()
    extractor.fit(df_train)

    # 3. Transform
    data_objects["train"] = extractor.transform(df_train)
    data_objects["val"] = extractor.transform(df_val)
    data_objects["test"] = extractor.transform(df_test)

    # 4. Save to Cache
    print("[DataProcessing] Saving to cache...")
    for split, data in data_objects.items():
        for key, value in data.items():
            ext = ".npy"
            to_save = value

            if key in ["lexical", "community"]:
                ext = ".npz"
                # Handle sparse matrix saving with provided utils
                if hasattr(value, "tocsr"):
                    # Deconstruct sparse matrix to dict of arrays for np.savez_compressed
                    value = value.tocsr()
                    to_save = {
                        "data": value.data,
                        "indices": value.indices,
                        "indptr": value.indptr,
                        "shape": np.array(value.shape),
                    }
                else:
                    # It's already something else?
                    to_save = {"arr": value}

            filename = f"X_{split}_{key}{ext}"
            utils.save_cache(to_save, filename)

    return data_objects["train"], data_objects["val"], data_objects["test"]
