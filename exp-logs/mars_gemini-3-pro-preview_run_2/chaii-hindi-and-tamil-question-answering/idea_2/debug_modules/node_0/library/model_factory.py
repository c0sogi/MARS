import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
from library.config import Config


def get_model(model_name_or_path=None):
    """
    Initializes and returns the Question Answering model.

    This function loads a pre-trained model (specifically MuRIL as defined in Config)
    with a Question Answering head. It can load from the Hugging Face Hub or
    a local directory containing a saved checkpoint.

    Args:
        model_name_or_path (str, optional): The model checkpoint name or local path.
                                            If None, defaults to Config.model_checkpoint.

    Returns:
        model: A Hugging Face AutoModelForQuestionAnswering instance.
    """
    if model_name_or_path is None:
        model_name_or_path = Config.model_checkpoint

    # Load configuration
    # This is useful if we want to inspect or modify config parameters before loading the model
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Load the model
    # AutoModelForQuestionAnswering will automatically add the classification head
    # (linear layer on top of hidden states) for start/end logits.
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path, config=config
    )

    return model


def get_tokenizer(model_name_or_path=None):
    """
    Initializes and returns the tokenizer corresponding to the model.

    Args:
        model_name_or_path (str, optional): The model checkpoint name or local path.
                                            If None, defaults to Config.model_checkpoint.

    Returns:
        tokenizer: A Hugging Face AutoTokenizer instance.
    """
    if model_name_or_path is None:
        model_name_or_path = Config.model_checkpoint

    # Load the tokenizer
    # We use fast=True (default in newer transformers) for better performance
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return tokenizer
