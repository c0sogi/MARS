import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name, pretrained=True):
    """
    Instantiates a PyTorch model using the timm library, adapted for binary classification.

    This function loads a backbone architecture (e.g., ConvNeXt or Swin Transformer)
    and modifies its final classification head to output a single logit, suitable for
    binary classification with BCEWithLogitsLoss.

    Args:
        model_name (str): The name of the model architecture to instantiate.
                          Must be a valid model name in the timm library
                          (e.g., 'convnext_small.fb_in22k', 'swin_small_patch4_window7_224').
        pretrained (bool): If True, loads weights pretrained on ImageNet (or ImageNet-22k).
                           Defaults to True.

    Returns:
        nn.Module: The instantiated model with a modified head for binary classification.
    """
    try:
        # Create the model using timm
        # num_classes=Config.NUM_CLASSES (1) ensures the head is replaced
        # with a Linear layer outputting 1 dimension.
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES
        )

        return model

    except Exception as e:
        print(f"Error creating model '{model_name}': {e}")
        # Re-raise the exception after logging to ensure the pipeline stops on failure
        raise e
