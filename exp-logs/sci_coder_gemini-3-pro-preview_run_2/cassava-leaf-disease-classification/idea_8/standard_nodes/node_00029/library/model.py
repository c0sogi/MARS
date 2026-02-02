import torch
import timm
from timm.utils import ModelEmaV2
from library.config import Config


def get_model(
    model_name=Config.model_name,
    num_classes=Config.num_classes,
    pretrained=True,
    use_ema=Config.use_ema,
):
    """
    Creates the model architecture and optionally initializes the Exponential Moving Average (EMA) wrapper.

    Args:
        model_name (str): The name of the model architecture to create via timm.
                          Defaults to Config.model_name.
        num_classes (int): The number of target classes.
                           Defaults to Config.num_classes.
        pretrained (bool): Whether to load pretrained ImageNet weights.
                           Defaults to True.
        use_ema (bool): Whether to initialize and return a ModelEmaV2 wrapper.
                        Defaults to Config.use_ema.

    Returns:
        tuple: (model, model_ema)
            - model (torch.nn.Module): The instantiated PyTorch model on the configured device.
            - model_ema (timm.utils.ModelEmaV2 or None): The EMA model wrapper if use_ema is True, else None.
    """

    # Create the model using timm
    # We inject regularization parameters (dropout and stochastic depth) from Config here.
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=Config.drop_rate,
        drop_path_rate=Config.drop_path_rate,
    )

    # Move the model to the computation device (GPU/CPU)
    model = model.to(Config.device)

    # Initialize Model EMA if requested
    # ModelEmaV2 maintains a moving average of model parameters which often leads to
    # better generalization and stability.
    model_ema = None
    if use_ema:
        model_ema = ModelEmaV2(
            model,
            decay=Config.model_ema_decay,
            device=Config.device,
        )

    return model, model_ema
