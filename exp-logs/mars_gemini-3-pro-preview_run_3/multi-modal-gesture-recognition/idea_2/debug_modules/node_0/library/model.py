import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualLayer(nn.Module):
    """
    A single residual block with dilated convolution.
    Structure: Dilated Conv -> ReLU -> 1x1 Conv -> Dropout -> Residual Add
    """

    def __init__(self, dilation, in_channels, out_channels, kernel_size, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding to keep the temporal dimension constant.
        # For a kernel size k and dilation d, the receptive field expansion is (k-1)*d.
        # We pad equally on both sides (non-causal) to maintain length.
        padding = int((kernel_size - 1) * dilation / 2)

        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        # If input and output channels differ, project input for residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(out + residual)


class SingleStageTCN(nn.Module):
    """
    A single stage of the MS-TCN.
    Consists of an input projection, a stack of dilated residual layers,
    and an output projection.
    """

    def __init__(
        self, input_dim, num_classes, num_layers, num_f_maps, kernel_size, dropout
    ):
        super(SingleStageTCN, self).__init__()

        self.conv_1x1_in = nn.Conv1d(input_dim, num_f_maps, 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                DilatedResidualLayer(
                    dilation=dilation,
                    in_channels=num_f_maps,
                    out_channels=num_f_maps,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        self.conv_1x1_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_1x1_out(out)
        return out


class MSTCN(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network.
    Stage 1: Prediction Network (Features -> Class Logits)
    Stage 2+: Refinement Networks (Class Probabilities -> Class Logits)
    """

    def __init__(self):
        super(MSTCN, self).__init__()

        self.stages = nn.ModuleList()

        # Stage 1: Prediction Stage
        # Takes raw input features (Input Dim) -> Outputs Class Logits
        self.stages.append(
            SingleStageTCN(
                input_dim=Config.INPUT_DIM,
                num_classes=Config.NUM_CLASSES,
                num_layers=Config.NUM_LAYERS,
                num_f_maps=Config.NUM_F_MAPS,
                kernel_size=Config.KERNEL_SIZE,
                dropout=Config.DROPOUT,
            )
        )

        # Subsequent Stages: Refinement Stages
        # Takes probabilities from previous stage (Num Classes) -> Outputs Class Logits
        for _ in range(Config.NUM_STAGES - 1):
            self.stages.append(
                SingleStageTCN(
                    input_dim=Config.NUM_CLASSES,
                    num_classes=Config.NUM_CLASSES,
                    num_layers=Config.NUM_LAYERS,
                    num_f_maps=Config.NUM_F_MAPS,
                    kernel_size=Config.KERNEL_SIZE,
                    dropout=Config.DROPOUT,
                )
            )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (Batch, Time, Features)
            mask: Optional mask of shape (Batch, Time). Not used in Conv1d computation
                  but accepted for compatibility with training loop.

        Returns:
            outputs: List of tensors, one for each stage.
                     Each tensor has shape (Batch, Classes, Time).
        """
        # Permute to (Batch, Features, Time) for Conv1d
        x = x.permute(0, 2, 1)

        outputs = []

        # Forward pass through Stage 1
        out = self.stages[0](x)
        outputs.append(out)

        # Forward pass through subsequent stages
        for stage in self.stages[1:]:
            # Input to refinement stage is softmax probabilities of previous stage
            probs = F.softmax(out, dim=1)
            out = stage(probs)
            outputs.append(out)

        return outputs
