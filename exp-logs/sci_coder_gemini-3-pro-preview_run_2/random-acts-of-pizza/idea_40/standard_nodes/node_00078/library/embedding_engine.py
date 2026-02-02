import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import process_with_cache, setup_logger


class EmbeddingEngine:
    """
    Manages the generation and caching of sentence embeddings for different views
    (Title, Body, Global Context) using pre-trained Transformer models.
    """

    def __init__(self):
        self.logger = setup_logger(
            "EmbeddingEngine", "./working/idea_40/embedding_engine.log"
        )
        self.models = {}

    def _get_model(self, model_name: str) -> SentenceTransformer:
        """
        Lazily loads and returns the requested SentenceTransformer model.
        """
        if model_name not in self.models:
            self.logger.info(f"Loading model: {model_name} on {Config.DEVICE}")
            self.models[model_name] = SentenceTransformer(
                model_name, device=Config.DEVICE
            )
        return self.models[model_name]

    def _compute_embeddings(self, texts: list, model_name: str) -> np.ndarray:
        """
        Encodes a list of texts into embeddings using the specified model.
        This function is intended to be passed to process_with_cache.
        """
        self.logger.info(
            f"Computing embeddings for {len(texts)} texts using {model_name}..."
        )
        model = self._get_model(model_name)

        # Ensure inputs are strings
        clean_texts = [str(t) if t is not None else "" for t in texts]

        # Encode
        embeddings = model.encode(
            clean_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # Normalization happens in the training pipeline
        )
        return embeddings

    def generate_train_embeddings(
        self, df_train: pd.DataFrame, load_cached_data: bool = True
    ):
        """
        Generates or loads embeddings for the training set.
        Returns a tuple: (title_embeddings, body_embeddings, global_embeddings)
        """
        self.logger.info("Generating/Loading Training Embeddings...")

        # 1. Title View (MiniLM)
        title_texts = df_train[Config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
        train_title_emb = process_with_cache(
            cache_path=Config.CACHE_TRAIN_TITLE_MINILM,
            process_fn=self._compute_embeddings,
            load_cached_data=load_cached_data,
            save_format="npy",
            texts=title_texts,
            model_name=Config.MODEL_MINILM,
        )

        # 2. Body View (MiniLM)
        body_texts = df_train[Config.TEXT_COL_BODY].fillna("").astype(str).tolist()
        train_body_emb = process_with_cache(
            cache_path=Config.CACHE_TRAIN_BODY_MINILM,
            process_fn=self._compute_embeddings,
            load_cached_data=load_cached_data,
            save_format="npy",
            texts=body_texts,
            model_name=Config.MODEL_MINILM,
        )

        # 3. Global Context View (MPNet) - Concatenation
        # We concatenate title and body with a space separator
        global_texts = (
            df_train[Config.TEXT_COL_TITLE].fillna("").astype(str)
            + " "
            + df_train[Config.TEXT_COL_BODY].fillna("").astype(str)
        ).tolist()

        train_global_emb = process_with_cache(
            cache_path=Config.CACHE_TRAIN_GLOBAL_MPNET,
            process_fn=self._compute_embeddings,
            load_cached_data=load_cached_data,
            save_format="npy",
            texts=global_texts,
            model_name=Config.MODEL_MPNET,
        )

        return train_title_emb, train_body_emb, train_global_emb

    def generate_test_embeddings(
        self, df_test: pd.DataFrame, load_cached_data: bool = True
    ):
        """
        Generates or loads embeddings for the test set.
        Returns a tuple: (title_embeddings, body_embeddings, global_embeddings)
        """
        self.logger.info("Generating/Loading Test Embeddings...")

        # 1. Title View (MiniLM)
        title_texts = df_test[Config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
        test_title_emb = process_with_cache(
            cache_path=Config.CACHE_TEST_TITLE_MINILM,
            process_fn=self._compute_embeddings,
            load_cached_data=load_cached_data,
            save_format="npy",
            texts=title_texts,
            model_name=Config.MODEL_MINILM,
        )

        # 2. Body View (MiniLM)
        body_texts = df_test[Config.TEXT_COL_BODY].fillna("").astype(str).tolist()
        test_body_emb = process_with_cache(
            cache_path=Config.CACHE_TEST_BODY_MINILM,
            process_fn=self._compute_embeddings,
            load_cached_data=load_cached_data,
            save_format="npy",
            texts=body_texts,
            model_name=Config.MODEL_MINILM,
        )

        # 3. Global Context View (MPNet) - Concatenation
        global_texts = (
            df_test[Config.TEXT_COL_TITLE].fillna("").astype(str)
            + " "
            + df_test[Config.TEXT_COL_BODY].fillna("").astype(str)
        ).tolist()

        test_global_emb = process_with_cache(
            cache_path=Config.CACHE_TEST_GLOBAL_MPNET,
            process_fn=self._compute_embeddings,
            load_cached_data=load_cached_data,
            save_format="npy",
            texts=global_texts,
            model_name=Config.MODEL_MPNET,
        )

        return test_title_emb, test_body_emb, test_global_emb
