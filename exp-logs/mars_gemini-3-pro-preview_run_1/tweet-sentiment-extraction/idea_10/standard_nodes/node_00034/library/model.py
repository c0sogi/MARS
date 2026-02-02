import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class SentimentConditionedDeberta(nn.Module):
    """
    DeBERTa-v3 with Sentiment-Conditioned Convolutional Head.

    Architecture:
    1. Backbone: microsoft/deberta-v3-base
    2. Pooling: Weighted Layer Pooling (last 4 layers)
    3. Conditioning: Late Fusion of Sentiment Token (from final layer) + Pooled Text
    4. Head: 1D Convolution -> Linear
    """

    def __init__(self):
        super(SentimentConditionedDeberta, self).__init__()
        # Load Configuration and Model Backbone
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_PATH, output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(Config.MODEL_PATH, config=self.config)

        # =====================================================================
        # Weighted Layer Pooling
        # =====================================================================
        # We aggregate the last 4 layers
        self.num_pool_layers = 4
        # Learnable weights for the layers, initialized to 1 (equal weight)
        self.layer_weights = nn.Parameter(torch.ones(self.num_pool_layers))

        # =====================================================================
        # Sentiment-Conditioned Convolutional Head
        # =====================================================================
        # Input: Pooled Representation (Hidden) + Sentiment Vector (Hidden)
        # Total Input Channels = 2 * Hidden Size
        self.conv = nn.Conv1d(
            in_channels=self.config.hidden_size * 2,
            out_channels=self.config.hidden_size,
            kernel_size=3,
            padding=1,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Final Projection to Start/End Logits
        self.fc = nn.Linear(self.config.hidden_size, 2)

        # Initialize custom weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for the custom head layers.
        """
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

        self.fc.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.fc.bias is not None:
            self.fc.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs [Batch, SeqLen]
            attention_mask (torch.Tensor): Attention mask [Batch, SeqLen]

        Returns:
            start_logits (torch.Tensor): Logits for start index [Batch, SeqLen]
            end_logits (torch.Tensor): Logits for end index [Batch, SeqLen]
        """
        # 1. Backbone Forward Pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 2. Weighted Layer Pooling
        # Stack the last 4 layers: [Batch, SeqLen, Hidden, 4]
        stacked_layers = torch.stack(all_hidden_states[-self.num_pool_layers :], dim=-1)

        # Calculate softmax weights to ensure they sum to 1
        weights = torch.softmax(self.layer_weights, dim=0)

        # Weighted sum: [Batch, SeqLen, Hidden]
        # Broadcast weights: [1, 1, 1, 4]
        pooled_output = (stacked_layers * weights.view(1, 1, 1, -1)).sum(dim=-1)

        # 3. Late Fusion Conditioning
        # Extract sentiment token embedding from the FINAL encoder layer (index -1)
        # Based on data.py, format is [CLS] sentiment [SEP] text ...
        # So sentiment token is at index 1.
        sentiment_vector = all_hidden_states[-1][:, 1, :]  # [Batch, Hidden]

        # Replicate sentiment vector across sequence length
        seq_len = input_ids.size(1)
        sentiment_repeated = sentiment_vector.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [Batch, SeqLen, Hidden]

        # Concatenate Pooled Text + Sentiment: [Batch, SeqLen, 2*Hidden]
        combined_features = torch.cat([pooled_output, sentiment_repeated], dim=-1)

        # 4. Conditioned Convolution
        # Permute for Conv1d: [Batch, Channels, SeqLen]
        x = combined_features.permute(0, 2, 1)

        # Apply Conv1d -> ReLU -> Dropout
        x = self.conv(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Permute back: [Batch, SeqLen, Channels]
        x = x.permute(0, 2, 1)

        # 5. Projection
        logits = self.fc(x)  # [Batch, SeqLen, 2]

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
