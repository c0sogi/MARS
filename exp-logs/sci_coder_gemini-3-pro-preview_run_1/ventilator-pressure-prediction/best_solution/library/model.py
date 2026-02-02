import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FeedForward(nn.Module):
    """
    Pointwise Feed-Forward Network (Linear -> GELU -> Linear).
    Adds depth and non-linearity to the state transition.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class InceptionStem(nn.Module):
    """
    Multi-Scale Stem using parallel 1D Convolutions with different kernel sizes.
    Captures both fine-grained signal noise and smoothed trend derivatives.
    """

    def __init__(self, input_dim, filters, kernels):
        super(InceptionStem, self).__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            # Padding = 'same' logic: (k - 1) // 2 for odd kernels
            padding = (k - 1) // 2
            self.convs.append(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=filters,
                    kernel_size=k,
                    padding=padding,
                )
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        # Conv1d expects: (Batch, Input_Dim, Seq_Len)
        x = x.permute(0, 2, 1)

        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))

        # Concatenate along channel dimension
        out = torch.cat(outputs, dim=1)
        out = self.activation(out)

        # Permute back to (Batch, Seq_Len, Filters * len(kernels))
        return out.permute(0, 2, 1)


class ContextInjectedResBlock(nn.Module):
    """
    Bidirectional LSTM Block with Deep Context Injection, Additive Residuals,
    and an internal Feed-Forward Network (FFN) block.
    Structure: x -> [LSTM+Res] -> [FFN+Res] -> out
    """

    def __init__(self, input_dim, hidden_size, context_dim, dropout_p=0.0):
        super(ContextInjectedResBlock, self).__init__()

        # The LSTM takes the concatenation of the signal input and context features
        self.lstm_input_dim = input_dim + context_dim
        self.hidden_size = hidden_size
        self.output_dim = hidden_size * 2  # Bidirectional

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Projection for residual connection if dimensions mismatch
        if input_dim != self.output_dim:
            self.projection = nn.Linear(input_dim, self.output_dim)
        else:
            self.projection = nn.Identity()

        # Feed-Forward Block
        self.ffn = FeedForward(
            input_dim=self.output_dim,
            hidden_dim=self.output_dim * Config.FFN_EXPANSION,
            dropout=dropout_p,
        )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, context):
        """
        Args:
            x: Signal input from previous layer (Batch, Seq, Input_Dim)
            context: Static context features (Batch, Seq, Context_Dim)
        """
        # --- Sub-layer 1: LSTM ---
        # Context Injection
        lstm_input = torch.cat([x, context], dim=-1)

        # LSTM Forward
        lstm_out, _ = self.lstm(lstm_input)

        # Residual Connection for LSTM
        # Note: We project x to match LSTM output dimension if necessary
        x_projected = self.projection(x)
        x = x_projected + self.dropout(lstm_out)

        # --- Sub-layer 2: Feed-Forward ---
        # FFN Forward
        ffn_out = self.ffn(x)

        # Residual Connection for FFN
        out = x + self.dropout(ffn_out)

        return out


class DeepSupervisedVentilatorModel(nn.Module):
    """
    Deeply Supervised Physics-Injected Residual Multi-Scale CNN-LSTM.
    """

    def __init__(self):
        super(DeepSupervisedVentilatorModel, self).__init__()

        # 1. Setup Dimensions & Indices
        self.input_dim = Config.INPUT_DIM
        self.hidden_size = Config.HIDDEN_SIZE
        self.context_features = Config.CONTEXT_FEATURES
        self.all_features = Config.FEATURE_COLS

        # Identify indices of context features within the input tensor
        self.context_indices = [
            i for i, col in enumerate(self.all_features) if col in self.context_features
        ]
        self.context_dim = len(self.context_indices)

        # 2. Multi-Scale Stem
        # We determine filter size to project roughly to Hidden Size
        kernels = Config.CNN_KERNELS
        filters_per_kernel = self.hidden_size // len(kernels)
        self.stem = InceptionStem(
            input_dim=self.input_dim, filters=filters_per_kernel, kernels=kernels
        )
        self.stem_out_dim = filters_per_kernel * len(kernels)

        # 3. Deep Context-Injected Backbone
        self.layers = nn.ModuleList()

        # Layer 1: Takes Stem Output
        self.layers.append(
            ContextInjectedResBlock(
                input_dim=self.stem_out_dim,
                hidden_size=self.hidden_size,
                context_dim=self.context_dim,
                dropout_p=Config.DROPOUT,
            )
        )

        # Layers 2-4: Take Previous Layer Output (Hidden * 2)
        for _ in range(Config.NUM_LAYERS - 1):
            self.layers.append(
                ContextInjectedResBlock(
                    input_dim=self.hidden_size * 2,
                    hidden_size=self.hidden_size,
                    context_dim=self.context_dim,
                    dropout_p=Config.DROPOUT,
                )
            )

        # 4. Heads
        # Auxiliary Head (after Layer 2, which is index 1)
        self.aux_head = nn.Linear(self.hidden_size * 2, 1)

        # Final Head (after Layer 4)
        self.final_head = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Seq_Len, Input_Dim)
        Returns:
            If training: (final_pred, aux_pred)
            If inference: final_pred
        """
        # Extract Context Features
        # x is (Batch, Seq, Features)
        context = x[:, :, self.context_indices]

        # Pass through Stem
        h = self.stem(x)

        # Pass through LSTM Layers
        # Layer 1
        h = self.layers[0](h, context)

        # Layer 2
        h = self.layers[1](h, context)

        # Deep Supervision: Capture Aux Output
        aux_out = self.aux_head(h)

        # Layer 3
        h = self.layers[2](h, context)

        # Layer 4
        h = self.layers[3](h, context)

        # Final Output
        final_out = self.final_head(h)

        # Squeeze last dimension to match target shape (Batch, Seq)
        final_out = final_out.squeeze(-1)
        aux_out = aux_out.squeeze(-1)

        if self.training:
            return final_out, aux_out
        else:
            return final_out
