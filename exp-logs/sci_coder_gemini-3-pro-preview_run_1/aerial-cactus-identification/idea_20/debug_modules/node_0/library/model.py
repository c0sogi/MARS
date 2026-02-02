import torch
import torch.nn as nn
from library.utils import RepVGG, RepVGGBlock, repvgg_model_convert

# Expose RepVGGBlock as requested by the module description
RepVGGBlock = RepVGGBlock


class RepVGGClassifier(RepVGG):
    """
    Deeply Supervised RepVGG Classifier.

    Architecture:
    - Conservative Stem: 3x3 Conv, Stride 1 (Preserves 32x32 resolution).
    - Deep Supervision: Auxiliary Classification Head attached after Stage 2.
    - Structural Re-parameterization: Multi-branch blocks (3x3 + 1x1 + Identity)
      during training, fusible to single 3x3 Conv for inference.

    Args:
        num_classes (int): Number of output classes (default: 1 for binary classification).
        width_multiplier (list): Multipliers for channel width at each stage.
                                 Default [1, 1, 1, 2] provides a balanced capacity.
        deploy (bool): If True, builds the model in inference mode (single branch).
                       If False, builds in training mode (multi-branch + aux head).
    """

    def __init__(self, num_classes=1, width_multiplier=[1, 1, 1, 2], deploy=False):
        super().__init__(
            num_classes=num_classes, width_multiplier=width_multiplier, deploy=deploy
        )


def model_to_deploy(model, save_path=None):
    """
    Performs Structural Re-parameterization to convert a training-mode model
    into an inference-mode model.

    This function fuses the 3x3, 1x1, and Identity branches of every RepVGGBlock
    into a single 3x3 Convolutional layer, and removes the auxiliary head.

    Args:
        model (nn.Module): The trained RepVGGClassifier in training mode.
        save_path (str, optional): Path to save the state_dict of the converted model.

    Returns:
        nn.Module: The converted model in deploy mode.
    """
    return repvgg_model_convert(model, save_path=save_path)
