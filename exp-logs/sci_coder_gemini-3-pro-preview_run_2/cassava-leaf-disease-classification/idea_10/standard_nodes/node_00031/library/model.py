import torch
import torch.nn as nn
import timm
from timm.utils import ModelEmaV2
from library.config import Config

# Mapping from Config.MODEL_BACKBONE to specific timm model names
TIMM_MODEL_MAP = {"convnext_small_in22k": "convnext_small.fb_in22k_ft_in1k"}


def get_model(
    model_name: str = Config.MODEL_BACKBONE,
    num_classes: int = Config.NUM_CLASSES,
    pretrained: bool = True,
    drop_path_rate: float = Config.DROP_PATH_RATE,
    use_ema: bool = Config.USE_EMA,
    checkpoint_path: str = None,
):
    """
    Constructs the model architecture and optionally the EMA wrapper.

    Args:
        model_name (str): The backbone identifier defined in Config.
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to initialize with ImageNet weights.
        drop_path_rate (float): Probability for Stochastic Depth (Drop Path).
        use_ema (bool): Whether to wrap the model with Exponential Moving Average.
        checkpoint_path (str, optional): Path to a .pth file to load weights from.

    Returns:
        tuple: (model, ema_model)
            - model (nn.Module): The instantiated PyTorch model on the configured device.
            - ema_model (ModelEmaV2 or None): The EMA wrapper if use_ema is True, else None.
    """

    # Resolve the specific timm model name
    timm_name = TIMM_MODEL_MAP.get(model_name, model_name)

    # Create the model using timm
    # This handles downloading pretrained weights and modifying the head for num_classes
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    # Move the model to the computation device (GPU/CPU)
    model = model.to(Config.DEVICE)

    # Initialize Exponential Moving Average (EMA) if requested
    # ModelEmaV2 requires the model to be on the device for correct initialization
    ema_model = None
    if use_ema:
        ema_model = ModelEmaV2(model, decay=Config.EMA_DECAY, device=Config.DEVICE)

    # Load weights from a checkpoint if a path is provided
    if checkpoint_path:
        # Load to CPU first to manage memory, then map to device
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

        # Identify the state dictionary within the checkpoint file
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if the model was saved using DataParallel
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load weights into the main model
        model.load_state_dict(state_dict, strict=True)

        # If EMA is active, attempt to load EMA specific states
        if use_ema and ema_model is not None:
            if "state_dict_ema" in checkpoint:
                ema_state_dict = checkpoint["state_dict_ema"]
                ema_state_dict = {
                    k.replace("module.", ""): v for k, v in ema_state_dict.items()
                }
                ema_model.load_state_dict(ema_state_dict)
            else:
                # If no EMA state is preserved in checkpoint, initialize EMA from the loaded model weights
                ema_model.set(model)

    return model, ema_model
