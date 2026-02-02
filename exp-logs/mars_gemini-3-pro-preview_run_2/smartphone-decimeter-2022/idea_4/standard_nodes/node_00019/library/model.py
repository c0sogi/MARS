import torch
import torch.nn as nn


class WindowedMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        window_size,
        fc_hidden=[256, 128, 64],
        dropout=0.2,
        output_dim=2,
        **kwargs
    ):
        """
        Simple Windowed MLP Smoother.

        Flattens the input window and processes it through dense layers.
        Simpler models often outperform deep sequences on sparse trajectory data.
        Cite solution_lesson_node_00010.

        Args:
            input_dim (int): Number of input features per timestep.
            window_size (int): Temporal size of the input window.
            fc_hidden (list): List of hidden units for the MLP.
            dropout (float): Dropout probability.
            output_dim (int): Dimension of the output.
        """
        super(WindowedMLP, self).__init__()

        self.flattened_dim = input_dim * window_size

        layers = []
        curr_dim = self.flattened_dim

        for hidden_dim in fc_hidden:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, context=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input window tensor. Shape: (Batch, Input_Dim, Window_Size)
            context: Ignored (kept for compatibility with engine loop).
        """
        # Flatten: (Batch, Input_Dim * Window_Size)
        # Note: x comes in as (Batch, Features, Time), we flatten to (Batch, Features*Time)
        x = x.view(x.size(0), -1)
        return self.net(x)
