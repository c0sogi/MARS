import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch

from library.config import Config
from library.utils import (
    get_logger,
    Timer,
    save_cache_npz,
    load_cache_npz,
    save_cache_npy,
    load_cache_npy,
    save_model,
    load_model,
)
from library.data_loader import load_and_preprocess_data

logger = get_logger("FeatureEngineering")


class LatentUserClusterer:
    """
    Performs the Latent User Clustering pipeline:
    Subreddit History -> TF-IDF -> SVD -> K-Means -> Centroid Distances.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(**Config.SUBREDDIT_TFIDF_PARAMS)
        self.svd = TruncatedSVD(
            n_components=Config.SVD_COMPONENTS, random_state=Config.RANDOM_STATE
        )
        self.kmeans = KMeans(
            n_clusters=Config.N_CLUSTERS,
            random_state=Config.RANDOM_STATE,
            n_init=10,
        )

    def _preprocess(self, subreddit_series):
        """Joins list of subreddits into space-separated strings."""
        return subreddit_series.apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )

    def fit(self, subreddit_series):
        with Timer("LatentUserClusterer Fit"):
            processed_data = self._preprocess(subreddit_series)
            tfidf_matrix = self.tfidf.fit_transform(processed_data)
            svd_matrix = self.svd.fit_transform(tfidf_matrix)
            self.kmeans.fit(svd_matrix)
        return self

    def transform(self, subreddit_series):
        processed_data = self._preprocess(subreddit_series)
        tfidf_matrix = self.tfidf.transform(processed_data)
        svd_matrix = self.svd.transform(tfidf_matrix)
        # Returns distances to all centroids (dense features)
        return self.kmeans.transform(svd_matrix)


class TextEmbedder:
    """
    Generates dense embeddings for text using a pre-trained Transformer model.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = Config.EMBEDDING_MODEL_NAME

    def transform(self, text_series):
        with Timer("TextEmbedder Transform"):
            # Load model only during transform to save memory when not in use
            model = SentenceTransformer(self.model_name, device=self.device)

            # Encode
            embeddings = model.encode(
                text_series.tolist(),
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return embeddings


class SparseFeaturizer:
    """
    Generates sparse TF-IDF features for both Text and Subreddit history.
    """

    def __init__(self):
        self.text_tfidf = TfidfVectorizer(**Config.TEXT_TFIDF_PARAMS)
        self.community_tfidf = TfidfVectorizer(**Config.SUBREDDIT_TFIDF_PARAMS)

    def _preprocess_subreddits(self, subreddit_series):
        return subreddit_series.apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )

    def fit(self, text_series, subreddit_series):
        with Timer("SparseFeaturizer Fit"):
            self.text_tfidf.fit(text_series)
            self.community_tfidf.fit(self._preprocess_subreddits(subreddit_series))
        return self

    def transform(self, text_series, subreddit_series):
        text_features = self.text_tfidf.transform(text_series)
        community_features = self.community_tfidf.transform(
            self._preprocess_subreddits(subreddit_series)
        )
        return {
            "lexical": text_features,
            "behavioral": community_features,
        }


class MetadataSelector:
    """
    Selects, concatenates, and scales numerical metadata and latent persona features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.numerical_cols = Config.NUMERICAL_COLS

    def fit(self, df, latent_features):
        with Timer("MetadataSelector Fit"):
            meta_part = df[self.numerical_cols].values
            # Concatenate metadata with latent cluster distances
            combined = np.hstack([meta_part, latent_features])
            self.scaler.fit(combined)
        return self

    def transform(self, df, latent_features):
        meta_part = df[self.numerical_cols].values
        combined = np.hstack([meta_part, latent_features])
        return self.scaler.transform(combined)


def get_features(load_cached_data=True):
    """
    Orchestrates the feature engineering pipeline.
    Checks cache first; if missing, computes features and saves them.

    Returns:
        dict: Nested dictionary containing features for 'train', 'val', 'test'.
              Structure:
              {
                  'train': {
                      'lexical': sparse_matrix,
                      'behavioral': sparse_matrix,
                      'semantic': numpy_array,
                      'metadata': numpy_array
                  },
                  ...
              }
    """
    # Define cache keys and filenames
    splits = ["train", "val", "test"]
    feature_types = {
        "lexical": "lexical.npz",
        "behavioral": "behavioral.npz",
        "semantic": "semantic.npy",
        "metadata": "metadata.npy",
    }

    # 1. Attempt to load from cache
    if load_cached_data:
        logger("Checking feature cache...")
        cached_data = {split: {} for split in splits}
        all_found = True

        for split in splits:
            for f_type, ext in feature_types.items():
                filename = f"X_{split}_{ext}"
                if ext.endswith(".npz"):
                    data = load_cache_npz(filename)
                    # npz might load as dict, we need the matrix if it's single key or just the object
                    # load_cache_npz returns NpzFile, we usually want the sparse matrix inside
                    # For scipy sparse saved with np.savez, it's tricky.
                    # Standard practice: use scipy.sparse.save_npz / load_npz.
                    # But utils provides save_cache_npz using np.savez.
                    # Let's assume we handle sparse matrices carefully.
                    # Actually, for sparse matrices, it's better to use scipy.sparse.save_npz.
                    # However, I must use the provided utils.
                    # Provided utils use np.savez. This is not ideal for sparse matrices unless decomposed.
                    # Wait, the utils `save_cache_npz` takes a dict. `load_cache_npz` returns a dict-like.
                    # If I save sparse matrices, I should probably densify or use a custom save.
                    # Given constraints, I will use the provided utils but I will assume
                    # I can save the components (data, indices, indptr, shape) if needed,
                    # OR I will simply use scipy.sparse.save_npz locally and bypass utils if utils is too rigid,
                    # BUT "You must import and use the functions...".
                    # Let's look at utils.py: `np.savez(filepath, **data_dict)`.
                    # This supports arrays. Sparse matrices are not arrays.
                    # I will convert sparse to a dict of arrays for storage using utils.
                    pass
                else:
                    data = load_cache_npy(filename)

                if data is None:
                    all_found = False
                    break
                cached_data[split][f_type] = data
            if not all_found:
                break

        # Re-assemble sparse matrices from loaded dicts if necessary
        if all_found:
            logger("All features found in cache. Reconstructing sparse matrices...")
            final_output = {}
            for split in splits:
                final_output[split] = {}
                for f_type, data in cached_data[split].items():
                    if f_type in ["lexical", "behavioral"]:
                        # Reconstruct sparse matrix from dict keys
                        # Assuming we saved: data, indices, indptr, shape
                        try:
                            mat = sp.csr_matrix(
                                (data["data"], data["indices"], data["indptr"]),
                                shape=data["shape"],
                            )
                            final_output[split][f_type] = mat
                        except Exception:
                            # Fallback if format is different
                            all_found = False
                            break
                    else:
                        final_output[split][f_type] = data

            if all_found:
                logger("Feature loading complete.")
                return final_output

    logger("Cache miss or force reload. Computing features from scratch...")

    # 2. Load Data
    train_df, val_df, test_df = load_and_preprocess_data(load_cached_data)

    # 3. Initialize Processors
    clusterer = LatentUserClusterer()
    embedder = TextEmbedder()
    sparse_featurizer = SparseFeaturizer()
    meta_selector = MetadataSelector()

    # 4. Fit on Training Data
    logger("Fitting LatentUserClusterer...")
    clusterer.fit(train_df[Config.SUBREDDIT_COL])

    logger("Fitting SparseFeaturizer...")
    sparse_featurizer.fit(train_df["text_concat"], train_df[Config.SUBREDDIT_COL])

    # 5. Transform and Generate Features
    datasets = {"train": train_df, "val": val_df, "test": test_df}
    output = {"train": {}, "val": {}, "test": {}}

    # We need latent features first as they are input to MetadataSelector
    latent_features = {}

    for split, df in datasets.items():
        logger(f"Processing split: {split}")

        # A. Latent User Clustering (Dense)
        logger(f"  - Generating Latent Persona Features ({split})...")
        latent = clusterer.transform(df[Config.SUBREDDIT_COL])
        latent_features[split] = latent

        # B. Text Embeddings (Dense)
        logger(f"  - Generating Text Embeddings ({split})...")
        output[split]["semantic"] = embedder.transform(df["text_concat"])

        # C. Sparse Features (Lexical + Behavioral)
        logger(f"  - Generating Sparse Features ({split})...")
        sparse_feats = sparse_featurizer.transform(
            df["text_concat"], df[Config.SUBREDDIT_COL]
        )
        output[split]["lexical"] = sparse_feats["lexical"]
        output[split]["behavioral"] = sparse_feats["behavioral"]

    # D. Metadata (Fit on Train, Transform All)
    logger("Fitting MetadataSelector...")
    meta_selector.fit(train_df, latent_features["train"])

    for split, df in datasets.items():
        logger(f"  - Generating Metadata ({split})...")
        output[split]["metadata"] = meta_selector.transform(df, latent_features[split])

    # 6. Save to Cache
    logger("Saving features to cache...")
    for split in splits:
        for f_type, data in output[split].items():
            filename = f"X_{split}_{feature_types[f_type]}"

            if sp.issparse(data):
                # Decompose sparse matrix for np.savez
                data_dict = {
                    "data": data.data,
                    "indices": data.indices,
                    "indptr": data.indptr,
                    "shape": data.shape,
                }
                save_cache_npz(data_dict, filename)
            else:
                save_cache_npy(data, filename)

    logger("Feature engineering complete.")
    return output
