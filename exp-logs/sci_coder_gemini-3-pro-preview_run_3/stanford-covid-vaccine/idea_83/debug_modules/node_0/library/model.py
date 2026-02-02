import torch
import torch.nn as nn
from library.config import Config
from library.layers import ConvStem, StabilizedGLUInteraction, RegressionHead


class DeepHierarchicalBiGRU(nn.Module):
    """
    Deep Hierarchical BiGRU with Deep Supervision and Stabilized GLU-Interaction.

    Architecture:
    1. ConvStem: Projects inputs to embedding space.
    2. Backbone (4 Layers):
       - BiGRU (384 hidden dim per direction -> 768 total)
       - Stabilized GLU-Decoupled Interaction (Structural Injection)
       - LayerNorm
    3. Deep Supervision:
       - A RegressionHead is attached to the output of EACH layer.
       - Returns a list of outputs [head1, head2, head3, head4].
    """

    def __init__(
        self,
        input_channels=Config.INPUT_CHANNELS,
        hidden_dim=Config.HIDDEN_DIM,  # 384
        num_layers=Config.NUM_LAYERS,  # 4
        dropout=Config.DROPOUT,
        num_targets=Config.NUM_TARGETS,
    ):
        super().__init__()
        self.num_layers = num_layers

        # 1. Convolutional Stem
        # Projects 14 channels -> 256 channels
        self.stem = ConvStem(input_channels=input_channels, kernel_size=3, filters=256)
        stem_output_dim = 256

        # 2. Sequential Blocks and Heads
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.heads = nn.ModuleList()

        # Total hidden dimension for BiGRU (Forward + Backward)
        rnn_output_dim = hidden_dim * 2

        for i in range(num_layers):
            # Input dim is stem output for first layer, else previous RNN output
            input_size = stem_output_dim if i == 0 else rnn_output_dim

            # A. BiGRU Layer
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # B. Stabilized GLU-Decoupled Interaction Module
            # Operates on the concatenated forward/backward states (size 768)
            interaction = StabilizedGLUInteraction(
                hidden_dim=rnn_output_dim, dropout=dropout
            )
            self.interaction_layers.append(interaction)

            # C. Layer Normalization
            norm = nn.LayerNorm(rnn_output_dim)
            self.norm_layers.append(norm)

            # D. Deep Supervision Head
            head = RegressionHead(input_dim=rnn_output_dim, output_dim=num_targets)
            self.heads.append(head)

    def forward(self, x, adjacency_indices):
        """
        Args:
            x: Input features (Batch, Seq_Len, 14)
            adjacency_indices: Paired indices (Batch, Seq_Len)

        Returns:
            outputs: List of tensors [out_layer1, out_layer2, ..., out_final]
                     Each tensor has shape (Batch, Seq_Len, 5)
        """
        # 1. Stem
        x = self.stem(x)

        outputs = []

        # 2. Process Blocks Sequentially
        for i in range(self.num_layers):
            # A. BiGRU
            # GRU returns (output, h_n), we only need output
            x, _ = self.gru_layers[i](x)

            # B. Interaction
            x = self.interaction_layers[i](x, adjacency_indices)

            # C. Normalization
            x = self.norm_layers[i](x)

            # D. Deep Supervision Head
            # We predict from the current layer's refined representation
            head_out = self.heads[i](x)
            outputs.append(head_out)

        # Return list of all head outputs for deep supervision loss calculation
        return outputs
