import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class LayerAggregator(nn.Module):
    """
    Computes a learnable weighted sum of the last 4 hidden states from the encoder.
    """

    def __init__(self):
        super().__init__()
        # Initialize weights for 4 layers.
        # Using a parameter that will be softmaxed ensures normalized weights.
        self.weights = nn.Parameter(torch.zeros(4))

    def forward(self, hidden_states):
        # hidden_states is a tuple of tensors from the backbone
        # We want the last 4 layers
        # Each tensor has shape (batch_size, seq_len, hidden_size)

        # Stack the last 4 layers along a new dimension
        # Shape: (batch_size, seq_len, hidden_size, 4)
        stacked_layers = torch.stack(hidden_states[-4:], dim=-1)

        # Compute softmax weights to ensure they sum to 1
        # Shape: (4,)
        norm_weights = torch.softmax(self.weights, dim=0)

        # Weighted sum along the last dimension
        # Broadcasting: (B, L, H, 4) * (4,) -> (B, L, H, 4)
        # Sum -> (B, L, H)
        weighted_sum = torch.sum(stacked_layers * norm_weights, dim=-1)

        return weighted_sum


class LinearAttentionPooling(nn.Module):
    """
    Implements Linear Attention Pooling: w^T h_t
    """

    def __init__(self, hidden_size):
        super().__init__()
        # Linear layer to compute attention scores (w^T * h_t)
        # Bias is False to strictly follow w^T h_t formulation
        self.attention_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # Compute scores: (batch_size, seq_len, 1)
        scores = self.attention_head(hidden_states)

        # Squeeze to (batch_size, seq_len)
        scores = scores.squeeze(-1)

        # Mask padding tokens with a large negative value so they have 0 attention weight
        # attention_mask is 1 for tokens, 0 for padding
        min_value = -1e9
        scores = scores.masked_fill(attention_mask == 0, min_value)

        # Compute attention weights via softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len)

        # Compute weighted sum of hidden states
        # Expand weights to (batch_size, seq_len, 1) for broadcasting
        attn_weights = attn_weights.unsqueeze(-1)

        # Context vector: sum(weights * states) -> (batch_size, hidden_size)
        context_vector = torch.sum(hidden_states * attn_weights, dim=1)

        return context_vector


class HybridPooling(nn.Module):
    """
    Combines Global Max Pooling and Linear Attention Pooling.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attn_pooling = LinearAttentionPooling(hidden_size)

    def forward(self, hidden_states, attention_mask):
        # 1. Global Max Pooling
        # We need to mask padding tokens so they don't affect the max
        # Expand mask: (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_size)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )

        # Set padding tokens to a large negative value
        hidden_states_masked = hidden_states.clone()
        hidden_states_masked[mask_expanded == 0] = -1e9

        # Max over sequence length dimension
        max_pooled = torch.max(hidden_states_masked, dim=1)[
            0
        ]  # (batch_size, hidden_size)

        # 2. Linear Attention Pooling
        attn_pooled = self.attn_pooling(
            hidden_states, attention_mask
        )  # (batch_size, hidden_size)

        # 3. Concatenate
        return torch.cat([max_pooled, attn_pooled], dim=1)


class ToxicityModel(nn.Module):
    """
    The main model architecture for Toxicity Classification.
    Features:
    - DeBERTa-v3-base backbone
    - Dynamic Layer Aggregation (last 4 layers)
    - Hybrid Pooling (Max + Linear Attention)
    - Multi-Sample Dropout
    """

    def __init__(
        self,
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        dropout_rate=Config.dropout,
    ):
        super().__init__()

        # Load Config and Model
        self.config = AutoConfig.from_pretrained(model_name)
        # Ensure we get hidden states for layer aggregation
        self.config.update({"output_hidden_states": True})

        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Custom Components
        self.aggregator = LayerAggregator()
        self.pooling = HybridPooling(self.config.hidden_size)

        # Multi-Sample Dropout
        # Create 5 parallel dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])

        # Shared Dense Layer
        # Input dimension is hidden_size * 2 because of Hybrid Pooling concatenation
        self.fc = nn.Linear(self.config.hidden_size * 2, num_classes)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Get hidden states (tuple of tensors)
        hidden_states = outputs.hidden_states

        # Aggregate last 4 layers
        feature_seq = self.aggregator(hidden_states)

        # Apply Hybrid Pooling
        pooled_output = self.pooling(feature_seq, attention_mask)

        # Multi-Sample Dropout & Prediction
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout
            dropped = dropout(pooled_output)
            # Apply shared FC
            logits = self.fc(dropped)
            logits_list.append(logits)

        # Average the logits
        # Stack: (5, batch_size, num_classes) -> Mean dim 0 -> (batch_size, num_classes)
        final_logits = torch.mean(torch.stack(logits_list), dim=0)

        return final_logits
