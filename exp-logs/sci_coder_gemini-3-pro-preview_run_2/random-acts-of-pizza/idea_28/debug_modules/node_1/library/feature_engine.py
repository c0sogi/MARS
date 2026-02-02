import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger, set_seed

logger = setup_logger("feature_engine")


class EmbeddingGenerator:
    """
    Generates and caches semantic embeddings for the Dual-Resolution Semantic Early Fusion strategy.
    Handles both High-Resolution (MiniLM) and Low-Resolution (MPNet) backbones.
    """

    def __init__(self):
        self.config = Config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_paths(self):
        """
        Resolves file paths for all embedding artifacts.
        Constructs validation paths locally as they are not explicitly in Config.
        """
        # Define paths for Val which are implicit in the workflow
        val_high_path = os.path.join(
            self.config.WORKING_DIR, "val_embeddings_high_res.npy"
        )
        val_low_path = os.path.join(
            self.config.WORKING_DIR, "val_embeddings_low_res.npy"
        )

        return {
            "high": {
                "train": self.config.TRAIN_EMBEDDINGS_HIGH_RES,
                "val": val_high_path,
                "test": self.config.TEST_EMBEDDINGS_HIGH_RES,
            },
            "low": {
                "train": self.config.TRAIN_EMBEDDINGS_LOW_RES,
                "val": val_low_path,
                "test": self.config.TEST_EMBEDDINGS_LOW_RES,
            },
        }

    def generate_embeddings(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates or loads raw embeddings for train, validation, and test sets.

        Args:
            train_df (pd.DataFrame): Training data with 'text_combined' column.
            val_df (pd.DataFrame): Validation data with 'text_combined' column.
            test_df (pd.DataFrame): Test data with 'text_combined' column.
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            tuple: (train_high, val_high, test_high, train_low, val_low, test_low)
                   All as numpy arrays.
        """
        set_seed()
        paths = self._get_paths()

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        # Check if all cache files exist
        missing_cache = False
        for res_type in ["high", "low"]:
            for split in ["train", "val", "test"]:
                if not os.path.exists(paths[res_type][split]):
                    missing_cache = True
                    break

        # Attempt to load from cache
        if load_cached_data and not missing_cache:
            logger.info("Loading embeddings from cache...")
            try:
                embeddings = {}
                for res_type in ["high", "low"]:
                    embeddings[res_type] = {}
                    for split in ["train", "val", "test"]:
                        embeddings[res_type][split] = np.load(paths[res_type][split])

                logger.info("Successfully loaded all embeddings.")
                return (
                    embeddings["high"]["train"],
                    embeddings["high"]["val"],
                    embeddings["high"]["test"],
                    embeddings["low"]["train"],
                    embeddings["low"]["val"],
                    embeddings["low"]["test"],
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load cache: {e}. Regenerating from scratch..."
                )

        logger.info("Generating embeddings from scratch...")

        # Prepare text data lists
        texts = {
            "train": train_df["text_combined"].tolist(),
            "val": val_df["text_combined"].tolist(),
            "test": test_df["text_combined"].tolist(),
        }

        results = {"high": {}, "low": {}}

        # ---------------------------------------------------------------------
        # 1. High Resolution (MiniLM)
        # ---------------------------------------------------------------------
        logger.info(f"Loading High-Res Model: {self.config.MODEL_HIGH_RES}")
        model_high = SentenceTransformer(self.config.MODEL_HIGH_RES, device=self.device)

        for split in ["train", "val", "test"]:
            logger.info(f"Encoding {split} set with High-Res model...")
            # Generate raw embeddings (normalization happens in the model pipeline)
            emb = model_high.encode(
                texts[split],
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            results["high"][split] = emb
            np.save(paths["high"][split], emb)

        # Cleanup to save memory
        del model_high
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # 2. Low Resolution (MPNet)
        # ---------------------------------------------------------------------
        logger.info(f"Loading Low-Res Model: {self.config.MODEL_LOW_RES}")
        model_low = SentenceTransformer(self.config.MODEL_LOW_RES, device=self.device)

        for split in ["train", "val", "test"]:
            logger.info(f"Encoding {split} set with Low-Res model...")
            # Generate raw embeddings (PCA and normalization happen in the model pipeline)
            emb = model_low.encode(
                texts[split],
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            results["low"][split] = emb
            np.save(paths["low"][split], emb)

        # Cleanup
        del model_low
        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Embeddings generation complete and saved.")

        return (
            results["high"]["train"],
            results["high"]["val"],
            results["high"]["test"],
            results["low"]["train"],
            results["low"]["val"],
            results["low"]["test"],
        )
