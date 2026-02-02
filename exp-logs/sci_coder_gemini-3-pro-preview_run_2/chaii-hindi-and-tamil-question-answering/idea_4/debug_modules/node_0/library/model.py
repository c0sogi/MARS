import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from library.config import Config


def get_tokenizer():
    """
    Loads and returns the tokenizer for the specific model architecture.

    Returns:
        transformers.PreTrainedTokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    return tokenizer


def get_model(pretrained=True):
    """
    Loads and returns the Question Answering model.

    Args:
        pretrained (bool): Whether to load pre-trained weights.
                           If False, loads configuration only (useful for testing/debugging).

    Returns:
        transformers.PreTrainedModel: The PyTorch model ready for training/inference.
    """
    config = AutoConfig.from_pretrained(Config.MODEL_NAME)

    if pretrained:
        model = AutoModelForQuestionAnswering.from_pretrained(
            Config.MODEL_NAME, config=config
        )
    else:
        model = AutoModelForQuestionAnswering.from_config(config)

    return model
