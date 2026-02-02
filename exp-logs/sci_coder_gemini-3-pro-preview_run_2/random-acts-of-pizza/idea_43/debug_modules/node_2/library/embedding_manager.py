import os
import numpy as np
import pandas as pd
import torch
import gc
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class EmbeddingManager:
    """
    Manages the computation, caching, and retrieval of multi-view embeddings
    for the HAMF-ADBE architecture.
    """

    def __init__(self):
        self.logger = setup_logger("embedding_manager")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64  # Efficient batch size for A100

    def _get_paths(self, split: str):
        """
        Returns the cache paths for the specified split.
        """
        if split == "train":
            return {
                "anchor_title": Config.CACHE_TRAIN_ANCHOR_TITLE,
                "anchor_body": Config.CACHE_TRAIN_ANCHOR_BODY,
                "aux_global": Config.CACHE_TRAIN_AUX_GLOBAL,
                "aux_hook": Config.CACHE_TRAIN_AUX_HOOK,
            }
        elif split == "test":
            return {
                "anchor_title": Config.CACHE_TEST_ANCHOR_TITLE,
                "anchor_body": Config.CACHE_TEST_ANCHOR_BODY,
                "aux_global": Config.CACHE_TEST_AUX_GLOBAL,
                "aux_hook": Config.CACHE_TEST_AUX_HOOK,
            }
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

    def _compute_and_save(self, model_name, sentences, save_path):
        """
        Helper to load a model, encode sentences, save to disk, and cleanup.
        """
        self.logger.info(f"Loading model {model_name} for embedding generation...")
        model = SentenceTransformer(model_name, device=self.device)

        self.logger.info(f"Encoding {len(sentences)} sentences...")
        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We handle normalization later in the pipeline
        )

        self.logger.info(f"Saving embeddings to {save_path}...")
        np.save(save_path, embeddings)

        # Cleanup to free GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return embeddings

    def get_embeddings(
        self, df: pd.DataFrame, split: str, load_cached_data: bool = True
    ):
        """
        Retrieves the four embedding views for the provided DataFrame.

        Args:
            df (pd.DataFrame): The dataframe containing text columns.
            split (str): 'train' or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (anchor_title, anchor_body, aux_global, aux_hook) as numpy arrays.
        """
        paths = self._get_paths(split)

        # Check if all cache files exist
        all_cached = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_cached:
            self.logger.info(f"Loading {split} embeddings from cache...")
            try:
                anchor_title = np.load(paths["anchor_title"])
                anchor_body = np.load(paths["anchor_body"])
                aux_global = np.load(paths["aux_global"])
                aux_hook = np.load(paths["aux_hook"])

                # Verify shapes match dataframe
                if len(anchor_title) == len(df):
                    self.logger.info("Cache loaded successfully.")
                    return anchor_title, anchor_body, aux_global, aux_hook
                else:
                    self.logger.warning(
                        f"Cache size mismatch (Cache: {len(anchor_title)}, DF: {len(df)}). Recomputing."
                    )
            except Exception as e:
                self.logger.warning(f"Error loading cache: {e}. Recomputing.")

        self.logger.info(f"Computing {split} embeddings from scratch...")

        # Prepare Text Data
        # Fill NaNs with empty string to prevent errors
        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str).tolist()

        # Create combined text for Global Context
        # Format: "Title [SEP] Body" (implicitly handled by space concatenation)
        global_texts = [t + " " + b for t, b in zip(titles, bodies)]

        # ---------------------------------------------------------
        # 1. Compute Anchor Embeddings (MiniLM)
        # ---------------------------------------------------------
        # We process both Title and Body with the Anchor model while it is loaded
        self.logger.info(f"Processing Anchor Views with {Config.ANCHOR_MODEL_NAME}...")
        model_anchor = SentenceTransformer(Config.ANCHOR_MODEL_NAME, device=self.device)

        # Title Anchor
        self.logger.info("Encoding Title Anchor...")
        anchor_title = model_anchor.encode(
            titles,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        np.save(paths["anchor_title"], anchor_title)

        # Body Anchor
        self.logger.info("Encoding Body Anchor...")
        anchor_body = model_anchor.encode(
            bodies,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        np.save(paths["anchor_body"], anchor_body)

        # Cleanup Anchor Model
        del model_anchor
        gc.collect()
        torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # 2. Compute Auxiliary Embeddings (MPNet)
        # ---------------------------------------------------------
        self.logger.info(f"Processing Auxiliary Views with {Config.AUX_MODEL_NAME}...")
        model_aux = SentenceTransformer(Config.AUX_MODEL_NAME, device=self.device)

        # Global Context (Title + Body)
        self.logger.info("Encoding Global Context...")
        aux_global = model_aux.encode(
            global_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        np.save(paths["aux_global"], aux_global)

        # Deep Hook (Title Only)
        self.logger.info("Encoding Deep Hook...")
        aux_hook = model_aux.encode(
            titles,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        np.save(paths["aux_hook"], aux_hook)

        # Cleanup Auxiliary Model
        del model_aux
        gc.collect()
        torch.cuda.empty_cache()

        self.logger.info(f"All {split} embeddings computed and saved.")

        return anchor_title, anchor_body, aux_global, aux_hook
