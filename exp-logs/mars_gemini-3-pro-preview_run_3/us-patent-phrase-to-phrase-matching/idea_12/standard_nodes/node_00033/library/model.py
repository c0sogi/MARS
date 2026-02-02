import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from library.config import CFG


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted sum of hidden states from all layers.
    Useful for combining low-level (lexical) and high-level (semantic) features.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 0):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        # Create a learnable weight for each layer to be used
        # +1 accounts for the embedding layer which is usually included in hidden_states
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # Stack the desired layers: (num_layers, batch_size, seq_len, hidden_size)
        hidden_states = torch.stack(all_hidden_states[self.layer_start :])

        # Calculate softmax-normalized weights
        weights = F.softmax(self.layer_weights, dim=0)

        # Reshape for broadcasting: (num_layers, 1, 1, 1)
        weights = weights.view(-1, 1, 1, 1)

        # Compute weighted sum across layers
        weighted_pooling_embeddings = (hidden_states * weights).sum(0)
        return weighted_pooling_embeddings


class AttentionPooling(nn.Module):
    """
    Aggregates token embeddings using an attention mechanism.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq, hidden)
        # attention_mask: (batch, seq)

        # Compute attention scores
        w = self.attention(last_hidden_state).float()  # (batch, seq, 1)

        # Mask padding tokens (set to -inf so softmax makes them 0)
        mask = attention_mask.unsqueeze(-1).expand(w.size())
        w[mask == 0] = float("-inf")

        # Normalize weights
        w = torch.softmax(w, dim=1)

        # Compute weighted sum of token embeddings
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class DebertaV3Model(nn.Module):
    """
    Main model class implementing DeBERTa-v3 with Dynamic Layer Mixing,
    Attention Pooling, and Multi-Sample Dropout.
    """

    def __init__(self, config=CFG, pretrained=True):
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(
            config.model_name, output_hidden_states=True
        )
        # Disable cache to prevent conflict with gradient checkpointing
        self.model_config.use_cache = False

        # Initialize Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(
                config.model_name, config=self.model_config
            )
        else:
            self.model = AutoModel.from_config(self.model_config)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Dynamic Layer Scalar Mixing
        if self.config.use_mix_layers:
            self.layer_pooling = WeightedLayerPooling(
                self.model_config.num_hidden_layers, layer_start=0
            )

        # Attention Pooling
        self.pooler = AttentionPooling(self.model_config.hidden_size)

        # Multi-Sample Dropout
        self.fc_dropout = nn.Dropout(config.fc_dropout)

        # Dual Heads
        # 1. Regression Head for Similarity Score
        self.fc_score = nn.Linear(self.model_config.hidden_size, config.target_size)

        # 2. Auxiliary Classification Head
        self.ce_bins = config.loss_config.get("ce_bins", 10)
        self.fc_logits = nn.Linear(self.model_config.hidden_size, self.ce_bins)

        # Initialize custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc_score)
        self._init_weights(self.fc_logits)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Backbone Forward Pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Layer Mixing
        if self.config.use_mix_layers:
            all_hidden_states = outputs.hidden_states
            feature = self.layer_pooling(all_hidden_states)
        else:
            feature = outputs.last_hidden_state

        # Pooling
        feature = self.pooler(feature, attention_mask)

        # Multi-Sample Dropout
        # Apply dropout and projection multiple times to average out noise
        score_sum = 0
        logits_sum = 0
        num_samples = 5

        for _ in range(num_samples):
            dropped_feature = self.fc_dropout(feature)
            score_sum += self.fc_score(dropped_feature)
            logits_sum += self.fc_logits(dropped_feature)

        final_score = score_sum / num_samples
        final_logits = logits_sum / num_samples

        return {"score": final_score, "logits": final_logits}
