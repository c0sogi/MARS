import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class EmbeddingGenerator:
    """
    Manages the generation of raw embeddings using pre-trained Transformer models.
    Implements caching to avoid redundant computations.
    """

    def __init__(self):
        """
        Initializes the EmbeddingGenerator with a logger and ensures the working directory exists.
        """
        self.logger = setup_logger(
            os.path.join(Config.WORKING_DIR, "feature_engine.log"),
            name="feature_engine",
        )
        self.device = Config.DEVICE

        # Ensure working directory exists for cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def _compute_embeddings(self, texts, model_name, batch_size=32):
        """
        Internal method to compute embeddings using SentenceTransformer.

        Args:
            texts (list): List of text strings to encode.
            model_name (str): Name of the pre-trained model.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Computed embeddings.
        """
        self.logger.info(f"Loading model: {model_name} on {self.device}")
        try:
            model = SentenceTransformer(model_name, device=self.device)

            # Ensure model is in eval mode
            model.eval()

            self.logger.info(
                f"Encoding {len(texts)} texts with batch_size={batch_size}..."
            )
            # normalize_embeddings=False because L2 norm is applied later per fold in the pipeline
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error computing embeddings for {model_name}: {e}")
            raise e

    def generate_embeddings(
        self, texts, model_name, cache_path, load_cached_data=True, batch_size=32
    ):
        """
        Generates embeddings for a list of texts, utilizing caching.

        Args:
            texts (list): List of text strings.
            model_name (str): Transformer model name.
            cache_path (str): Full path to the cache file (.npy).
            load_cached_data (bool): Whether to attempt loading from cache.
            batch_size (int): Batch size for inference.

        Returns:
            np.ndarray: The embeddings matrix.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached embeddings from {cache_path}")
            try:
                embeddings = np.load(cache_path)
                if embeddings.shape[0] == len(texts):
                    return embeddings
                else:
                    self.logger.warning(
                        f"Cached embeddings shape {embeddings.shape} does not match text count {len(texts)}. "
                        "Recomputing..."
                    )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cache {cache_path}: {e}. Recomputing..."
                )

        # 2. Compute from scratch
        self.logger.info(
            f"Computing embeddings for {model_name} (Cache miss or invalid)..."
        )
        embeddings = self._compute_embeddings(texts, model_name, batch_size)

        # 3. Save to cache
        self.logger.info(f"Saving embeddings to {cache_path}")
        try:
            np.save(cache_path, embeddings)
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_path}: {e}")

        return embeddings

    def generate_dataset_embeddings(
        self, df_train, df_val, df_test, load_cached_data=True, batch_size=32
    ):
        """
        Orchestrates the generation of embeddings for all datasets and all configured models.

        Args:
            df_train (pd.DataFrame): Training data.
            df_val (pd.DataFrame): Validation data.
            df_test (pd.DataFrame): Test data.
            load_cached_data (bool): Whether to use cached data.
            batch_size (int): Batch size for inference.

        Returns:
            dict: Nested dictionary containing embeddings for 'anchor' and 'aux' models for each split.
                  Structure: results[model_type][split_name] -> np.ndarray
        """
        tasks = [("anchor", Config.ANCHOR_MODEL_NAME), ("aux", Config.AUX_MODEL_NAME)]

        splits = [("train", df_train), ("val", df_val), ("test", df_test)]

        results = {}

        for model_key, model_name in tasks:
            self.logger.info(f"Processing {model_key} model: {model_name}")
            results[model_key] = {}

            for split_name, df in splits:
                # Construct cache filename: e.g., train_emb_anchor.npy
                filename = f"{split_name}_emb_{model_key}.npy"
                cache_path = os.path.join(Config.WORKING_DIR, filename)

                # Extract texts
                # data_loader.py creates 'text_combined' column
                if "text_combined" not in df.columns:
                    raise ValueError(
                        f"Column 'text_combined' missing from {split_name} dataframe."
                    )

                texts = df["text_combined"].astype(str).tolist()

                embeddings = self.generate_embeddings(
                    texts=texts,
                    model_name=model_name,
                    cache_path=cache_path,
                    load_cached_data=load_cached_data,
                    batch_size=batch_size,
                )

                results[model_key][split_name] = embeddings

        return results
