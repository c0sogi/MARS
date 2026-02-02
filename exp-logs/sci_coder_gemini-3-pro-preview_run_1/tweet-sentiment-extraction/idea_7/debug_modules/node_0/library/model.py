import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class TweetModel(nn.Module):
    """
    DeBERTa-v3 with Multi-Scale Context Aggregation Head.

    This model implements a span extraction architecture that refines boundary detection
    by aggregating context at multiple scales (token, local neighbors, phrase) and
    fusing features from the top layers of the transformer backbone.

    Architecture:
    1. Backbone: DeBERTa-v3-base
    2. Head:
       - Weighted Layer Pooling (last 4 layers)
       - Multi-Scale 1D Convolutions (k=1, 3, 5)
       - Concatenation -> Dropout -> Linear Projection
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Load configuration to determine hidden size
        model_config = AutoConfig.from_pretrained(config.MODEL_PATH)
        self.hidden_size = model_config.hidden_size

        # Initialize Backbone
        # We use output_hidden_states=True in forward, but loading it here ensures weights are correct
        self.deberta = AutoModel.from_pretrained(config.MODEL_PATH, config=model_config)

        # Weighted Layer Pooling: Learnable weights for the last 4 layers
        # Initialized to equal weights (1/4 each)
        self.layer_weights = nn.Parameter(torch.tensor([1 / 4] * 4))

        # Multi-Scale Context Aggregation Head
        # Parallel 1D Convolutions with different kernel sizes
        # k=1: Captures token-specific features
        # k=3: Captures immediate local context
        # k=5: Captures wider phrase-level context
        # Padding is set to (kernel_size - 1) // 2 to maintain sequence length
        self.conv1 = nn.Conv1d(
            self.hidden_size, self.hidden_size, kernel_size=1, padding=0
        )
        self.conv3 = nn.Conv1d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv1d(
            self.hidden_size, self.hidden_size, kernel_size=5, padding=2
        )

        # Regularization
        self.dropout = nn.Dropout(0.1)

        # Final Projection: Concatenated output of 3 scales -> 2 logits (start, end)
        self.fc = nn.Linear(self.hidden_size * 3, 2)

        # Initialize weights for the custom head
        self._init_weights()

    def _init_weights(self):
        """
        Apply Xavier Uniform initialization to the final linear layer.
        """
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (Batch, Seq_Len)
            attention_mask (torch.Tensor): Attention mask (Batch, Seq_Len)

        Returns:
            start_logits (torch.Tensor): Logits for start index (Batch, Seq_Len)
            end_logits (torch.Tensor): Logits for end index (Batch, Seq_Len)
        """
        # Get backbone outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Stack the last 4 hidden states: (4, Batch, Seq, Hidden)
        all_hidden_states = torch.stack(outputs.hidden_states[-4:])

        # Apply Weighted Layer Pooling
        # Softmax ensures weights sum to 1 across the layer dimension
        weights = F.softmax(self.layer_weights, dim=0).view(4, 1, 1, 1)
        pooled_output = (weights * all_hidden_states).sum(dim=0)  # (Batch, Seq, Hidden)

        # Permute for Conv1d: (Batch, Hidden, Seq)
        x = pooled_output.permute(0, 2, 1)

        # Apply Multi-Scale Convolutions with ReLU activation
        c1 = F.relu(self.conv1(x))
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))

        # Concatenate features from different scales: (Batch, Hidden*3, Seq)
        cat = torch.cat([c1, c3, c5], dim=1)

        # Permute back to sequence format: (Batch, Seq, Hidden*3)
        cat = cat.permute(0, 2, 1)

        # Apply Dropout
        cat = self.dropout(cat)

        # Project to logits: (Batch, Seq, 2)
        logits = self.fc(cat)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (Batch, Seq)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
