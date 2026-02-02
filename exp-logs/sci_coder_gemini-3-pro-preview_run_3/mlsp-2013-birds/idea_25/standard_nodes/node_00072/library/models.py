import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    Bird Species Classification Model using heterogeneous backbones.
    Wraps timm models to provide a consistent interface and supports
    ResNet-18, EfficientNet-B0, and DenseNet-121.
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        """
        Args:
            backbone_name (str): Name of the backbone (e.g., 'resnet18', 'efficientnet_b0').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdModel, self).__init__()
        self.backbone_name = backbone_name

        # Create the model using timm
        # timm handles loading pretrained weights and replacing the classification head
        # when num_classes is specified.
        self.model = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


def get_llrd_params(
    model: BirdModel, base_lr: float, weight_decay: float, decay_rate: float
):
    """
    Groups model parameters for Layer-Wise Learning Rate Decay (LLRD).

    The learning rate for a parameter group is calculated as:
    LR = base_lr * (decay_rate ^ depth_from_head)

    Args:
        model (BirdModel): The model instance.
        base_lr (float): The base learning rate for the head (classifier).
        weight_decay (float): Weight decay for regularization.
        decay_rate (float): The multiplicative decay factor per layer/block.

    Returns:
        list: A list of dictionaries suitable for the optimizer.
    """
    backbone_name = model.backbone_name
    timm_model = model.model

    # We will collect groups of parameters starting from the stem (input) to the head (output).
    # Later, we will reverse this list to assign decay rates (Head gets base_lr).
    groups = []

    if "resnet" in backbone_name:
        # ResNet Structure (timm):
        # conv1, bn1 -> layer1 -> layer2 -> layer3 -> layer4 -> fc

        # Group 0: Stem
        stem_params = []
        if hasattr(timm_model, "conv1"):
            stem_params.extend(list(timm_model.conv1.parameters()))
        if hasattr(timm_model, "bn1"):
            stem_params.extend(list(timm_model.bn1.parameters()))
        groups.append(stem_params)

        # Groups 1-4: Stages
        groups.append(list(timm_model.layer1.parameters()))
        groups.append(list(timm_model.layer2.parameters()))
        groups.append(list(timm_model.layer3.parameters()))
        groups.append(list(timm_model.layer4.parameters()))

        # Group 5: Head
        head_params = []
        if hasattr(timm_model, "fc"):
            head_params.extend(list(timm_model.fc.parameters()))
        if hasattr(timm_model, "global_pool"):
            head_params.extend(list(timm_model.global_pool.parameters()))
        groups.append(head_params)

    elif "densenet" in backbone_name:
        # DenseNet Structure (timm):
        # features (conv0, norm0, denseblock1...4, transition1...3) -> classifier
        features = timm_model.features

        # Group 0: Stem
        stem_params = []
        stem_params.extend(list(features.conv0.parameters()))
        stem_params.extend(list(features.norm0.parameters()))
        groups.append(stem_params)

        # Groups 1-4: DenseBlocks + Transitions
        # Block 1
        g1 = list(features.denseblock1.parameters()) + list(
            features.transition1.parameters()
        )
        groups.append(g1)

        # Block 2
        g2 = list(features.denseblock2.parameters()) + list(
            features.transition2.parameters()
        )
        groups.append(g2)

        # Block 3
        g3 = list(features.denseblock3.parameters()) + list(
            features.transition3.parameters()
        )
        groups.append(g3)

        # Block 4 + Final Norm
        g4 = list(features.denseblock4.parameters()) + list(features.norm5.parameters())
        groups.append(g4)

        # Group 5: Head
        groups.append(list(timm_model.classifier.parameters()))

    elif "efficientnet" in backbone_name:
        # EfficientNet Structure (timm):
        # conv_stem, bn1 -> blocks (0..6) -> conv_head, bn2 -> classifier

        # Group 0: Stem + Block 0
        g0 = (
            list(timm_model.conv_stem.parameters())
            + list(timm_model.bn1.parameters())
            + list(timm_model.blocks[0].parameters())
        )
        groups.append(g0)

        # Groups 1-5: Blocks 1-5
        groups.append(list(timm_model.blocks[1].parameters()))
        groups.append(list(timm_model.blocks[2].parameters()))
        groups.append(list(timm_model.blocks[3].parameters()))
        groups.append(list(timm_model.blocks[4].parameters()))
        groups.append(list(timm_model.blocks[5].parameters()))

        # Group 6: Block 6 + Top Conv/BN
        g6 = (
            list(timm_model.blocks[6].parameters())
            + list(timm_model.conv_head.parameters())
            + list(timm_model.bn2.parameters())
        )
        groups.append(g6)

        # Group 7: Head
        groups.append(list(timm_model.classifier.parameters()))

    else:
        # Fallback for generic models: Split into Backbone and Head
        head_names = ["fc", "classifier", "head"]
        head_params = []
        backbone_params = []

        head_found = False
        for name, param in model.named_parameters():
            is_head = False
            for hname in head_names:
                if hname in name:
                    is_head = True
                    break

            if is_head:
                head_params.append(param)
                head_found = True
            else:
                backbone_params.append(param)

        if not head_found:
            groups.append(list(model.parameters()))
        else:
            groups.append(backbone_params)
            groups.append(head_params)

    # Reverse groups to assign Learning Rates from Head (max) to Stem (min)
    # groups[0] is now the Head
    groups_reversed = groups[::-1]

    optimizer_params = []

    for i, group_params in enumerate(groups_reversed):
        if len(group_params) == 0:
            continue

        # LR = base_lr * (decay_rate ^ depth)
        lr = base_lr * (decay_rate**i)

        optimizer_params.append(
            {"params": group_params, "lr": lr, "weight_decay": weight_decay}
        )

    return optimizer_params
