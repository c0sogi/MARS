import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class MultiLayerFusion(nn.Module):
    """
    Combines the last N layers of the backbone using a learnable weighted sum.
    """

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(Config.fusion_layers))

    def forward(self, hidden_states):
        # hidden_states: tuple of (batch_size, seq_len, hidden_size)
        # We expect the input to be the last Config.fusion_layers states

        # Stack to shape: (batch_size, seq_len, hidden_size, num_layers)
        stacked = torch.stack(hidden_states, dim=-1)

        # Calculate softmax weights: (num_layers,)
        weights = F.softmax(self.weights, dim=0)

        # Weighted sum along the last dimension
        # Reshape weights to (1, 1, 1, num_layers) for broadcasting
        fused = torch.sum(stacked * weights.view(1, 1, 1, -1), dim=-1)

        return fused


class LinearAttentionPooling(nn.Module):
    """
    Linear Attention Pooling: w^T * h_t
    Computes a weighted average of the sequence outputs based on a learned scoring vector.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, hidden_size)
        # mask: (batch_size, seq_len)

        # Compute scores: (batch_size, seq_len, 1)
        scores = self.attention(x)

        # Mask padding tokens (set score to -inf)
        # mask is 1 for tokens, 0 for padding
        min_value = -1e9
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        scores = scores.masked_fill(mask_expanded == 0, min_value)

        # Compute attention weights
        att_weights = F.softmax(scores, dim=1)

        # Weighted sum: (batch_size, hidden_size)
        context = torch.sum(att_weights * x, dim=1)

        return context


class ToxicModel(nn.Module):
    """
    Domain-Adapted DeBERTa-v3 with Multi-Layer Fusion, Hybrid Pooling, and Multi-Sample Dropout.
    """

    def __init__(self, name_or_path=None):
        super().__init__()

        # Use provided path (e.g., DAPT weights) or default from Config
        model_path = name_or_path if name_or_path else Config.model_name

        # Load configuration
        config = AutoConfig.from_pretrained(model_path)
        config.output_hidden_states = True

        # Disable internal dropout to rely on Multi-Sample Dropout for regularization
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0

        # Initialize Backbone
        self.backbone = AutoModel.from_pretrained(model_path, config=config)

        # Initialize Fusion Layer
        self.fusion = MultiLayerFusion()

        # Initialize Pooling Layer
        self.attention_pool = LinearAttentionPooling(Config.hidden_size)

        # Classification Head
        # Input dimension is hidden_size * 2 because we concatenate:
        # 1. Global Max Pooling (hidden_size)
        # 2. Linear Attention Pooling (hidden_size)
        self.fc = nn.Linear(Config.hidden_size * 2, Config.num_labels)

        # Initialize weights for custom layers
        self._init_weights(self.fc)
        self._init_weights(self.attention_pool.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Backbone Forward Pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract Hidden States
        # outputs.hidden_states is a tuple of (batch, seq, hidden) for all layers + embedding
        all_hidden_states = outputs.hidden_states

        # Select the last N layers for fusion
        states_to_fuse = all_hidden_states[-Config.fusion_layers :]

        # Apply Multi-Layer Fusion
        fused_sequence = self.fusion(states_to_fuse)  # (batch, seq, hidden)

        # --- Hybrid Pooling ---

        # 1. Global Max Pooling
        # Mask padding tokens to -inf so they don't affect max
        mask_expanded = attention_mask.unsqueeze(-1)
        fused_masked = fused_sequence.masked_fill(mask_expanded == 0, -1e9)
        max_pooled = torch.max(fused_masked, dim=1)[0]  # (batch, hidden)

        # 2. Linear Attention Pooling
        att_pooled = self.attention_pool(
            fused_sequence, attention_mask
        )  # (batch, hidden)

        # Concatenate
        pooled_output = torch.cat(
            [max_pooled, att_pooled], dim=1
        )  # (batch, hidden * 2)

        # --- Multi-Sample Dropout ---

        if Config.use_multi_sample_dropout:
            logits_list = []
            for _ in range(Config.dropout_samples):
                # Apply dropout with a fresh mask each time
                dropped = F.dropout(
                    pooled_output, p=Config.dropout_rate, training=self.training
                )
                logits_list.append(self.fc(dropped))

            # Average the logits from all dropout samples
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            dropped = F.dropout(
                pooled_output, p=Config.dropout_rate, training=self.training
            )
            logits = self.fc(dropped)

        return logits
