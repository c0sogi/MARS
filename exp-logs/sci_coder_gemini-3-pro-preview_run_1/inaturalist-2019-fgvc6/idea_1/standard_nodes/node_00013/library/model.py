import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
):
    """
    Constructs a model using timm.
    Cite solution_lesson_node_00011: Use better architectures/weights.
    Using tf_efficientnet_b0.ns_jft_in1k for superior pretraining.
    """
    print(f"Creating model: {Config.MODEL_NAME}")
    model = timm.create_model(
        Config.MODEL_NAME, pretrained=pretrained, num_classes=num_classes
    )

    # Move the model to the specified device
    model = model.to(device)

    return model
