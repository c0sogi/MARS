import torch
import torch.nn as nn


class ContextAware1DCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        context_dim,
        window_size,
        conv_channels=[32, 64, 128],
        kernel_size=3,
        fc_hidden=[256, 128],
        dropout=0.2,
        output_dim=2,
    ):
        """
        Context-Aware 1D-CNN Smoother.

        The model takes a window of sensor data and a context vector (absolute state)
        to predict the residual correction for the center timestamp.

        Args:
            input_dim (int): Number of input features per timestep (channels).
            context_dim (int): Number of context features (e.g., WLS Lat, Lon, Alt).
            window_size (int): Temporal size of the input window.
            conv_channels (list): List of output channels for each convolutional layer.
            kernel_size (int): Kernel size for 1D convolutions.
            fc_hidden (list): List of hidden units for the MLP head.
            dropout (float): Dropout probability.
            output_dim (int): Dimension of the output (2 for DeltaEast, DeltaNorth).
        """
        super(ContextAware1DCNN, self).__init__()

        # ---------------------------------------------------------
        # 1. Convolutional Backbone
        # ---------------------------------------------------------
        # We use padding to keep the sequence length constant through the layers.
        # This allows the dense layers to see the full temporal context.
        padding = (kernel_size - 1) // 2

        layers = []
        in_channels = input_dim

        for out_channels in conv_channels:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)

        # Calculate flattened dimension after convolution
        # Since padding='same' (effectively), length remains window_size
        self.flattened_dim = in_channels * window_size

        # ---------------------------------------------------------
        # 2. Context-Aware Prediction Head (MLP)
        # ---------------------------------------------------------
        # The input to the MLP is the concatenated vector of:
        # - Flattened temporal features (from CNN)
        # - Absolute context features (from WLS baseline)
        head_input_dim = self.flattened_dim + context_dim

        head_layers = []
        curr_dim = head_input_dim

        for hidden_dim in fc_hidden:
            head_layers.append(nn.Linear(curr_dim, hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim

        # Final projection to output targets
        head_layers.append(nn.Linear(curr_dim, output_dim))
        self.head = nn.Sequential(*head_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming Normal for ReLU networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, context):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input window tensor.
                              Shape: (Batch, Input_Dim, Window_Size)
            context (torch.Tensor): Absolute state context tensor.
                                    Shape: (Batch, Context_Dim)

        Returns:
            torch.Tensor: Predicted residuals.
                          Shape: (Batch, Output_Dim)
        """
        # 1. Extract temporal features
        # Shape: (Batch, Last_Conv_Channel, Window_Size)
        features = self.backbone(x)

        # 2. Flatten
        # Shape: (Batch, Flattened_Dim)
        features = features.view(features.size(0), -1)

        # 3. Context Injection
        # Concatenate flattened temporal features with absolute context
        # Shape: (Batch, Flattened_Dim + Context_Dim)
        combined = torch.cat([features, context], dim=1)

        # 4. Predict
        # Shape: (Batch, Output_Dim)
        out = self.head(combined)

        return out
