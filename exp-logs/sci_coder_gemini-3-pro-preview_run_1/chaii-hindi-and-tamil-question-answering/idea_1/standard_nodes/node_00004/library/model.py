import os
import torch
from transformers import MT5ForConditionalGeneration, logging
from library.config import Config

# Suppress verbose warning messages from the transformers library
logging.set_verbosity_error()


def load_model(model_path=None, device=None):
    """
    Initializes and loads the MT5ForConditionalGeneration model.

    This function serves as a factory for the model, allowing for loading
    either the base pre-trained model defined in Config or a specific
    checkpoint from a local path.

    Args:
        model_path (str, optional): The model identifier (e.g., 'google/mt5-small')
                                    or a local path to a saved model directory.
                                    If None, defaults to Config.MODEL_NAME.
        device (torch.device, optional): The device to move the model to.
                                         If None, defaults to Config.DEVICE.

    Returns:
        MT5ForConditionalGeneration: The loaded model instance on the specified device.
    """
    # Determine the model path to use
    if model_path is None:
        model_path = Config.MODEL_NAME

    # Determine the device to use
    if device is None:
        device = Config.DEVICE

    # Load the model using the transformers library
    # MT5ForConditionalGeneration is used for the sequence-to-sequence QA task
    try:
        model = MT5ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        raise OSError(f"Error loading model from path '{model_path}': {e}")

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    return model
