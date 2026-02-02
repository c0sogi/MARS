import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class EmbeddingGenerator:
    """
    Handles the generation and caching of sentence embeddings for the
    Whitened Multi-Field Asymmetric Dual-Backbone Ensemble.
    """

    def __init__(self):
        self.logger = setup_logger("EmbeddingGenerator")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define paths for validation embeddings (not explicitly in Config)
        self.val_anchor_path = os.path.join(
            Config.WORKING_DIR, "val_anchor_embeddings.npy"
        )
        self.val_aux_path = os.path.join(Config.WORKING_DIR, "val_aux_embeddings.npy")

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def generate_embeddings(self, train_df, val_df, test_df, load_cached: bool = True):
        """
        Generates or loads embeddings for Train, Validation, and Test sets.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.
            load_cached (bool): If True, attempts to load from disk first.

        Returns:
            dict: Dictionary containing numpy arrays for all views and splits.
                  Keys: 'train_anchor', 'val_anchor', 'test_anchor',
                        'train_aux', 'val_aux', 'test_aux'
        """
        # List of all required files
        required_files = [
            Config.TRAIN_ANCHOR_EMB_PATH,
            self.val_anchor_path,
            Config.TEST_ANCHOR_EMB_PATH,
            Config.TRAIN_AUX_EMB_PATH,
            self.val_aux_path,
            Config.TEST_AUX_EMB_PATH,
        ]

        # Check cache
        if load_cached and all(os.path.exists(f) for f in required_files):
            self.logger.info("All embedding files found in cache. Loading...")
            return self._load_cache()

        self.logger.info("Generating embeddings from scratch...")

        # ==========================================
        # 1. Generate Anchor Embeddings (MiniLM)
        # ==========================================
        self.logger.info(f"Loading Anchor Model: {Config.ANCHOR_MODEL_NAME}")
        anchor_model = SentenceTransformer(Config.ANCHOR_MODEL_NAME, device=self.device)

        self.logger.info("Computing Anchor embeddings for Train set...")
        train_anchor = self._compute_anchor(anchor_model, train_df)

        self.logger.info("Computing Anchor embeddings for Validation set...")
        val_anchor = self._compute_anchor(anchor_model, val_df)

        self.logger.info("Computing Anchor embeddings for Test set...")
        test_anchor = self._compute_anchor(anchor_model, test_df)

        # Free memory
        del anchor_model
        torch.cuda.empty_cache()

        # ==========================================
        # 2. Generate Aux Embeddings (MPNet)
        # ==========================================
        self.logger.info(f"Loading Aux Model: {Config.AUX_MODEL_NAME}")
        aux_model = SentenceTransformer(Config.AUX_MODEL_NAME, device=self.device)

        self.logger.info("Computing Aux embeddings for Train set...")
        train_aux = self._compute_aux(aux_model, train_df)

        self.logger.info("Computing Aux embeddings for Validation set...")
        val_aux = self._compute_aux(aux_model, val_df)

        self.logger.info("Computing Aux embeddings for Test set...")
        test_aux = self._compute_aux(aux_model, test_df)

        # Free memory
        del aux_model
        torch.cuda.empty_cache()

        # ==========================================
        # 3. Save to Cache
        # ==========================================
        self.logger.info("Saving embeddings to disk...")
        np.save(Config.TRAIN_ANCHOR_EMB_PATH, train_anchor)
        np.save(self.val_anchor_path, val_anchor)
        np.save(Config.TEST_ANCHOR_EMB_PATH, test_anchor)

        np.save(Config.TRAIN_AUX_EMB_PATH, train_aux)
        np.save(self.val_aux_path, val_aux)
        np.save(Config.TEST_AUX_EMB_PATH, test_aux)

        return {
            "train_anchor": train_anchor,
            "val_anchor": val_anchor,
            "test_anchor": test_anchor,
            "train_aux": train_aux,
            "val_aux": val_aux,
            "test_aux": test_aux,
        }

    def _compute_anchor(self, model, df):
        """
        Computes Anchor embeddings: Concat(Encode(Title), Encode(Body))
        """
        # Encode Title
        titles = df["request_title"].fillna("").astype(str).tolist()
        title_emb = model.encode(
            titles, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # Encode Body
        bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()
        body_emb = model.encode(
            bodies, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # Concatenate horizontally: [Title_Emb, Body_Emb]
        # Shape: (N, 384) + (N, 384) -> (N, 768)
        return np.hstack([title_emb, body_emb])

    def _compute_aux(self, model, df):
        """
        Computes Aux embeddings: Encode(Concat(Title, Body))
        """
        # Use pre-concatenated text from DataLoader
        texts = df["text_concat"].fillna("").astype(str).tolist()

        # Shape: (N, 768)
        return model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

    def _load_cache(self):
        """
        Loads all embeddings from disk.
        """
        return {
            "train_anchor": np.load(Config.TRAIN_ANCHOR_EMB_PATH),
            "val_anchor": np.load(self.val_anchor_path),
            "test_anchor": np.load(Config.TEST_ANCHOR_EMB_PATH),
            "train_aux": np.load(Config.TRAIN_AUX_EMB_PATH),
            "val_aux": np.load(self.val_aux_path),
            "test_aux": np.load(Config.TEST_AUX_EMB_PATH),
        }
