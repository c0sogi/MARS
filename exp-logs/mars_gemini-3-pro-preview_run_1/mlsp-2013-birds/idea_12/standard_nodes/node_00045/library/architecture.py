import torch
import torch.nn as nn
import timm


def get_seresnet_model(config):
    """
    Initializes and returns the SE-ResNet-34 model.

    This function uses the `timm` library to create a ResNet-34 backbone augmented
    with Squeeze-and-Excitation (SE) blocks. It is configured to accept 3-channel
    inputs (channel replication of the spectrogram) and output logits for the
    specified number of classes via a simple linear head.

    Args:
        config (Config): Configuration object containing model hyperparameters
                         (MODEL_NAME, PRETRAINED, NUM_CLASSES, CHANNELS).

    Returns:
        nn.Module: The configured PyTorch model.
    """
    # Ensure the model name is consistent with the expectation
    model_name = config.MODEL_NAME  # Expected: 'seresnet34'

    # Create the model using timm
    # - pretrained=True: Loads weights pre-trained on ImageNet.
    # - num_classes=config.NUM_CLASSES: Replaces the default 1000-class head
    #   with a new Linear layer for the 19 bird species.
    # - in_chans=config.CHANNELS: Configures the first convolutional layer
    #   to accept 3 input channels (RGB).
    model = timm.create_model(
        model_name,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES,
        in_chans=config.CHANNELS,
    )

    return model
