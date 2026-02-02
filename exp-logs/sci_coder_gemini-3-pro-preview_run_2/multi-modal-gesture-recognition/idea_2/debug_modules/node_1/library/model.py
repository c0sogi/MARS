import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualLayer(nn.Module):
    """
    A single dilated convolutional layer with a residual connection.
    Structure: Dilated Conv1d -> ReLU -> Conv1d (1x1) -> Dropout -> Residual Add
    """

    def __init__(self, dilation, in_channels, out_channels, kernel_size=3, dropout=0.5):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding to maintain temporal dimension
        # For kernel_size=3, padding = dilation
        padding = dilation * (kernel_size - 1) // 2

        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    """
    A single stage of the TCN, consisting of a stack of DilatedResidualLayers.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageTCN, self).__init__()

        # Input projection
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        # Stack of dilated residual layers
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    dilation=2**i,
                    in_channels=num_f_maps,
                    out_channels=num_f_maps,
                    kernel_size=Config.KERNEL_SIZE,
                )
                for i in range(num_layers)
            ]
        )

        # Output projection to classes
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageTCN(nn.Module):
    """
    The full MS-TCN architecture.
    Stage 1: Prediction Stage (Features -> Logits)
    Stage 2+: Refinement Stages (Probabilities -> Logits)
    """

    def __init__(
        self,
        num_stages=Config.NUM_STAGES,
        num_layers=Config.NUM_LAYERS,
        num_f_maps=Config.NUM_F_MAPS,
        dim=Config.INPUT_DIM,
        num_classes=Config.NUM_CLASSES,
    ):
        super(MultiStageTCN, self).__init__()

        # Stage 1: Takes raw features (dim) as input
        self.stage1 = SingleStageTCN(num_layers, num_f_maps, dim, num_classes)

        # Subsequent Stages: Take class probabilities (num_classes) as input
        self.stages = nn.ModuleList(
            [
                SingleStageTCN(num_layers, num_f_maps, num_classes, num_classes)
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features of shape (Batch, Time, Dim)
            mask: Optional mask (Batch, Time) - not strictly used in forward convs
                  but kept for API consistency.
        Returns:
            outputs: List of tensors, one per stage. Each tensor is (Batch, Classes, Time).
        """
        # Permute to (Batch, Dim, Time) for Conv1d
        x = x.permute(0, 2, 1)

        outputs = []

        # Stage 1
        out = self.stage1(x)
        outputs.append(out)

        # Refinement Stages
        for stage in self.stages:
            # Input to next stage is softmax probabilities of current stage
            in_refinement = F.softmax(out, dim=1)
            out = stage(in_refinement)
            outputs.append(out)

        return outputs
