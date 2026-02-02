import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig
from library.config import Config


def get_tokenizer():
    """
    Loads the XLM-Roberta tokenizer based on the configuration.

    Returns:
        transformers.PreTrainedTokenizer: The loaded tokenizer.
    """
    # Load the tokenizer from the checkpoint defined in Config
    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)
    return tokenizer


def get_model(weights_path=None):
    """
    Loads the XLM-Roberta model for Question Answering.

    Args:
        weights_path (str, optional): Path to a directory containing a fine-tuned model.
                                      If None, loads the base pre-trained model from Config.

    Returns:
        transformers.PreTrainedModel: The loaded model moved to the configured device.
    """
    if weights_path and os.path.exists(weights_path):
        # Load fine-tuned model configuration and weights
        config = AutoConfig.from_pretrained(weights_path)
        model = AutoModelForQuestionAnswering.from_pretrained(
            weights_path, config=config
        )
    else:
        # Load base pre-trained model
        model = AutoModelForQuestionAnswering.from_pretrained(Config.model_checkpoint)

    # Move model to the computation device (GPU/CPU)
    model.to(Config.device)

    return model


def save_model(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer to the specified output directory.

    Args:
        model (transformers.PreTrainedModel): The model to save.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to save.
        output_dir (str): The directory path where artifacts should be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights and configuration
    model.save_pretrained(output_dir)

    # Save tokenizer files
    tokenizer.save_pretrained(output_dir)
