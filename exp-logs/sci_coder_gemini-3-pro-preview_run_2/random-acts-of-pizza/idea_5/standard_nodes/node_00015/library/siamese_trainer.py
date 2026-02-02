import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from library import config, data_handler, preprocessor, utils

logger = utils.setup_logger("siamese_trainer")


class FineTuner:
    """
    Encapsulates the fine-tuning logic for the representation learning stage.
    Uses a Siamese Network approach with BatchHardTripletLoss to optimize
    embeddings for the binary classification task.
    """

    def __init__(self):
        self.model_name = config.TRANSFORMER_MODEL_NAME
        self.save_path = config.FINE_TUNED_MODEL_PATH
        self.device = config.DEVICE
        self.model = None

    def train(self, load_cached_data: bool = True):
        """
        Setup the model.
        Cite Lesson 00014: On small datasets (N < 5,000), prefer frozen pre-trained embeddings over fine-tuning.
        We skip the fine-tuning process entirely to prevent manifold collapse and overfitting.
        """
        logger.info(
            "Skipping fine-tuning to use frozen pre-trained embeddings (Lesson 00014)."
        )
        self.load_model()

    def load_model(self):
        """
        Loads the base pre-trained model.
        """
        logger.info(f"Loading base model {self.model_name} (Frozen)...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list) -> object:
        """
        Generates embeddings for a list of texts using the current model.

        Args:
            texts (list): List of strings to encode.

        Returns:
            numpy.ndarray: Array of embeddings.
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Encoding {len(texts)} texts...")
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True,
        )
