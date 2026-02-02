import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling (Weighted Layer Pooling) mechanism.
    Learns a weight for each token's embedding to dynamically emphasize specific tokens.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            context_vector: (batch_size, hidden_size)
        """
        # Calculate attention scores
        # h = tanh(W * H)
        h = self.tanh(self.W(last_hidden_state))

        # score = V * h
        score = self.V(h).squeeze(-1)  # (batch_size, seq_len)

        # Mask padding tokens so they don't contribute to the weighted sum
        if attention_mask is not None:
            # Set score of padding tokens to a very small number
            score = score.masked_fill(attention_mask == 0, -1e9)

        # Calculate weights via Softmax
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Compute weighted sum of hidden states
        context_vector = torch.sum(
            last_hidden_state * weights, dim=1
        )  # (batch_size, hidden_size)

        return context_vector


class InsultModel(nn.Module):
    """
    Main model architecture for Insult Detection.
    Backbone: DeBERTa-v3-Large
    Head: Attention Pooling + Multi-Sample Dropout
    """

    def __init__(self):
        super(InsultModel, self).__init__()

        # Load Transformer Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.model_path)
        self.backbone = AutoModel.from_pretrained(Config.model_path, config=self.config)

        # Initialize Custom Components
        self.pooling = AttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout: Create a list of dropout layers with different rates
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.dropout_rates])

        # Final Classification Layer
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for custom layers
        self._init_weights(self.pooling)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for custom modules using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        Returns:
            logits: (batch_size, 1) - Raw scores before sigmoid
        """
        # Pass through Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = (
            outputs.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

        # Apply Attention Pooling
        feature = self.pooling(
            last_hidden_state, attention_mask
        )  # (batch_size, hidden_size)

        # Apply Multi-Sample Dropout
        # Pass the feature through multiple dropout masks and the same linear layer
        logits_list = []
        for dropout in self.dropouts:
            dropped_feature = dropout(feature)
            logits_list.append(self.fc(dropped_feature))

        # Average the logits from all dropout samples
        # Stack: (num_dropouts, batch_size, 1) -> Mean dim 0 -> (batch_size, 1)
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits
