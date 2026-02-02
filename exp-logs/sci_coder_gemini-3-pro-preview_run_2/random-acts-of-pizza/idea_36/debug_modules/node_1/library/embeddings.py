import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger, save_npy, load_npy, set_seed
from library.data_loader import load_dataset

# Initialize Logger
logger = setup_logger("embedding_generator")


class EmbeddingGenerator:
    """
    Manages the computation and caching of text embeddings for the
    Context-Aware Asymmetric Early Fusion strategy.
    """

    def __init__(self):
        """
        Initializes the generator and sets the random seed and device.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(Config.SEED)
        logger.info(f"EmbeddingGenerator initialized on device: {self.device}")

    def _resolve_config(self, split, backbone_type):
        """
        Resolves the model name and cache path based on split and backbone type.

        Args:
            split (str): 'train', 'val', or 'test'.
            backbone_type (str): 'anchor' or 'aux'.

        Returns:
            tuple: (model_name, cache_path)
        """
        if backbone_type == "anchor":
            model_name = Config.ANCHOR_MODEL_NAME
            if split == "train":
                cache_path = Config.TRAIN_EMB_ANCHOR_PATH
            elif split == "val":
                cache_path = Config.VAL_EMB_ANCHOR_PATH
            elif split == "test":
                cache_path = Config.TEST_EMB_ANCHOR_PATH
            else:
                raise ValueError(f"Unknown split: {split}")

        elif backbone_type == "aux":
            model_name = Config.AUX_MODEL_NAME
            if split == "train":
                cache_path = Config.TRAIN_EMB_AUX_PATH
            elif split == "val":
                cache_path = Config.VAL_EMB_AUX_PATH
            elif split == "test":
                cache_path = Config.TEST_EMB_AUX_PATH
            else:
                raise ValueError(f"Unknown split: {split}")

        else:
            raise ValueError(
                f"Unknown backbone_type: {backbone_type}. Must be 'anchor' or 'aux'."
            )

        return model_name, cache_path

    def _compute_embeddings(self, texts, model_name):
        """
        Loads the model and computes embeddings for the provided texts.

        Args:
            texts (list): List of text strings.
            model_name (str): Name of the SentenceTransformer model.

        Returns:
            np.ndarray: Computed embeddings.
        """
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, device=self.device)
        model.eval()

        logger.info(f"Encoding {len(texts)} texts...")
        # batch_size=32 is a safe default for these models on standard GPUs
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # Normalization happens in the training pipeline
        )

        # Clean up memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embeddings

    def get_embeddings(self, split, backbone_type, load_cached_data=True):
        """
        Retrieves embeddings for a specific split and backbone.
        Implements strict caching logic.

        Args:
            split (str): 'train', 'val', or 'test'.
            backbone_type (str): 'anchor' or 'aux'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: The embeddings matrix.
        """
        model_name, cache_path = self._resolve_config(split, backbone_type)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(
                f"Loading cached {backbone_type} embeddings for {split} from {cache_path}"
            )
            try:
                embeddings = load_npy(cache_path)
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        logger.info(f"Computing {backbone_type} embeddings for {split} from scratch...")

        # Load text data
        # We assume load_dataset handles its own caching of the dataframe
        df = load_dataset(split, load_cached_data=load_cached_data)
        texts = df["text_concat"].tolist()

        # Generate
        embeddings = self._compute_embeddings(texts, model_name)

        # Save to cache
        logger.info(f"Saving embeddings to {cache_path}")
        save_npy(embeddings, cache_path)

        return embeddings

    def generate_all(self, load_cached_data=True):
        """
        Convenience method to ensure all embeddings (train/val/test x anchor/aux) are generated.
        Useful for initialization steps.
        """
        splits = ["train", "val", "test"]
        backbones = ["anchor", "aux"]

        for split in splits:
            for backbone in backbones:
                self.get_embeddings(split, backbone, load_cached_data)
