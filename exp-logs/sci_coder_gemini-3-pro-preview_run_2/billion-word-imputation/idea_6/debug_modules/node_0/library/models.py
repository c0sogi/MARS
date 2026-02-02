import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoConfig,
)
from library.config import Config
from library.utils import get_logger

logger = get_logger("models")


class ModelFactory:
    """
    Factory class to instantiate the specific model architectures for the
    Probabilistic Beam-Search Cascade solution.
    """

    @staticmethod
    def get_locator_model():
        """
        Initializes the Locator model (Stage 1).

        Architecture: DeBERTa-v3 (Decoding-enhanced BERT with disentangled attention).
        Head: Token Classification (Linear layer on top of hidden states).

        Configuration:
            - Pretrained weights: microsoft/deberta-v3-base (from Config)
            - num_labels: 1 (Binary regression/classification per token: Gap vs No Gap)

        Returns:
            transformers.PreTrainedModel: The initialized DeBERTa model for token classification.
        """
        model_name = Config.LOCATOR_MODEL_NAME
        logger.info(f"Loading Locator model architecture from: {model_name}")

        try:
            # We use num_labels=1. The output will be of shape (batch_size, seq_len, 1).
            # During training, these logits are compared against binary targets (0/1)
            # using BCEWithLogitsLoss.
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, num_labels=1
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load Locator model: {e}")
            raise e

    @staticmethod
    def get_infiller_model():
        """
        Initializes the In-Filler model (Stage 2).

        Architecture: RoBERTa-Large.
        Head: Masked Language Modeling (MLM).

        Configuration:
            - Pretrained weights: roberta-large (from Config)

        Returns:
            transformers.PreTrainedModel: The initialized RoBERTa model for MLM.
        """
        model_name = Config.INFILLER_MODEL_NAME
        logger.info(f"Loading In-Filler model architecture from: {model_name}")

        try:
            # Standard MLM head initialization.
            # This model predicts the probability of tokens in the vocabulary for masked positions.
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            return model
        except Exception as e:
            logger.error(f"Failed to load In-Filler model: {e}")
            raise e
