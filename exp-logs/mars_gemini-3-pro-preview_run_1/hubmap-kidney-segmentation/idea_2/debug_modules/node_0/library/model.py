import torch
import segmentation_models_pytorch as smp
from library.config import Config


def build_model(
    encoder_name=Config.ENCODER,
    encoder_weights=Config.ENCODER_WEIGHTS,
    in_channels=Config.IN_CHANNELS,
    classes=Config.CLASSES,
    activation=Config.ACTIVATION,
):
    """
    Constructs the U-Net++ deep learning architecture with a specified encoder.

    Utilizes segmentation_models_pytorch to instantiate the model, handling
    the complex nested skip connections and pre-trained encoder initialization.

    Args:
        encoder_name (str): Name of the encoder backbone (e.g., 'efficientnet-b5').
        encoder_weights (str): Pre-trained weights to load (e.g., 'imagenet').
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        classes (int): Number of output classes (e.g., 1 for binary segmentation).
        activation (str or None): Activation function to apply to the output
                                  (e.g., 'sigmoid', 'softmax', or None for logits).

    Returns:
        torch.nn.Module: The constructed U-Net++ model.
    """

    # Instantiate the U-Net++ model
    # The library automatically downloads and loads the specified encoder weights
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )

    return model
