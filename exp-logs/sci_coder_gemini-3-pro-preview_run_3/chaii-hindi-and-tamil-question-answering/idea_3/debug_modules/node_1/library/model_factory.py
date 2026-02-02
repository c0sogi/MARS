import os
import torch
from transformers import (
    AutoTokenizer,
    XLMRobertaForMaskedLM,
    XLMRobertaForTokenClassification,
    AutoConfig,
)
from library.config import Config


def get_tokenizer():
    """
    Loads the Fast Tokenizer for XLM-RoBERTa.

    Returns:
        transformers.PreTrainedTokenizerFast: The tokenizer instance.
    """
    # use_fast=True is required for return_offsets_mapping in data processing
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    return tokenizer


def get_tapt_model():
    """
    Initializes the Masked Language Model for Task-Adaptive Pretraining (TAPT).
    Loads the base XLM-RoBERTa model with an MLM head.

    Returns:
        transformers.XLMRobertaForMaskedLM: The model for MLM training.
    """
    model = XLMRobertaForMaskedLM.from_pretrained(Config.MODEL_NAME)
    return model


def get_qa_model(model_path=None):
    """
    Initializes the Token Classification model for Question Answering.

    This function handles two scenarios:
    1. Initializing from base weights (if model_path is None).
    2. Initializing from TAPT-adapted weights (if model_path is provided).

    Args:
        model_path (str, optional): Path to a pretrained model directory or checkpoint.
                                    If None, defaults to Config.MODEL_NAME.

    Returns:
        transformers.XLMRobertaForTokenClassification: The model for QA fine-tuning.
    """
    if model_path is None:
        model_path = Config.MODEL_NAME

    # We explicitly set num_labels to 3 (O, B-ANS, I-ANS)
    # When loading from an MLM checkpoint (TAPT), the library will issue a warning
    # that the 'lm_head' weights are unused and the 'classifier' weights are newly initialized.
    # This is the expected behavior for transfer learning.
    model = XLMRobertaForTokenClassification.from_pretrained(
        model_path, num_labels=Config.NUM_LABELS
    )

    return model
