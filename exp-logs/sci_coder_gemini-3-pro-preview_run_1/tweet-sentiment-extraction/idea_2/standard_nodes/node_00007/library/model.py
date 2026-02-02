import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TweetModel(nn.Module):
    """
    RoBERTa-based model for Sentiment Extraction with Weighted Layer Pooling.

    Architecture:
    1. RoBERTa Backbone (roberta-base)
    2. Weighted Layer Pooling: Aggregates the last N hidden layers using learned weights.
    3. Dropout
    4. Linear Head: Predicts start and end logits for the selected text.
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load configuration and enable hidden states output
        config = AutoConfig.from_pretrained(Config.ROBERTA_PATH)
        config.output_hidden_states = True

        # Load pre-trained RoBERTa model
        self.roberta = AutoModel.from_pretrained(Config.ROBERTA_PATH, config=config)

        # Weighted Layer Pooling Parameters
        # We learn a scalar weight for each of the N layers to be aggregated.
        # Initializing to 0.0 results in equal weights after Softmax (1/N).
        self.layer_weights = nn.Parameter(torch.zeros(Config.N_LAST_HIDDEN))

        # Dropout for regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # Classification Head
        # Maps hidden_size (768) to 2 outputs (start_logit, end_logit)
        self.out = nn.Linear(Config.HIDDEN_SIZE, 2)

        # Initialize weights for the custom head
        self._init_weights(self.out)

    def _init_weights(self, module):
        """
        Initialize weights for the linear layer.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]

        Returns:
            start_logits (torch.Tensor): Logits for start position [batch_size, seq_len]
            end_logits (torch.Tensor): Logits for end position [batch_size, seq_len]
        """
        # Pass inputs through backbone
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # Extract hidden states
        # outputs.hidden_states is a tuple of tensors, each shape [batch_size, seq_len, hidden_size]
        all_hidden_states = outputs.hidden_states

        # Stack the last N layers designated in Config
        # Shape: [batch_size, n_layers, seq_len, hidden_size]
        stacked_layers = torch.stack(all_hidden_states[-Config.N_LAST_HIDDEN :], dim=1)

        # Compute normalized weights for the layers
        # Shape: [n_layers]
        weights = torch.softmax(self.layer_weights, dim=0)

        # Reshape weights for broadcasting: [1, n_layers, 1, 1]
        weights = weights.view(1, -1, 1, 1)

        # Compute weighted sum of hidden states
        # Shape: [batch_size, seq_len, hidden_size]
        weighted_output = (stacked_layers * weights).sum(dim=1)

        # Apply dropout
        x = self.dropout(weighted_output)

        # Pass through classification head
        # Shape: [batch_size, seq_len, 2]
        logits = self.out(x)

        # Split logits into start and end
        start_logits, end_logits = logits.split(1, dim=-1)

        # Remove the last dimension to get [batch_size, seq_len]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
