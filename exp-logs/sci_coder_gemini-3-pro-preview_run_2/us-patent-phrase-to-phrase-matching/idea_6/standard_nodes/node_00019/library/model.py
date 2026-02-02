import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import CFG


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to aggregate the sequence of hidden states into a single vector.
    Learns a weighted average of the token embeddings.
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
        w = self.attention(last_hidden_state)
        float_mask = attention_mask.unsqueeze(-1).float()
        w = w.masked_fill(float_mask == 0, -1e4)
        w = torch.softmax(w, dim=1)
        return torch.sum(w * last_hidden_state, dim=1)


class HybridDeberta(nn.Module):
    """
    Hybrid DeBERTa-v3-Large model that fuses semantic embeddings with structural features.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(CFG.model_name)
        self.config.attention_probs_dropout_prob = CFG.dropout
        self.config.hidden_dropout_prob = CFG.dropout

        if pretrained:
            self.model = AutoModel.from_pretrained(CFG.model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        self.pooler = AttentionPooling(self.config.hidden_size)

        # Structural features dimension: 3 (Normalized Levenshtein, Jaccard Similarity, Length Ratio)
        self.structural_dim = 3

        # Classification head: Hidden Size + Structural Features -> Target Size (5 classes)
        self.fc = nn.Linear(
            self.config.hidden_size + self.structural_dim, CFG.target_size
        )
        self.fc_dropout = nn.Dropout(CFG.fc_dropout)

        self._init_weights(self.pooler)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for custom layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, structural_features):
        """
        Forward pass of the hybrid model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            structural_features (torch.Tensor): Tensor of shape (batch_size, 3) containing
                                                Levenshtein, Jaccard, and Length Ratio features.

        Returns:
            torch.Tensor: Logits for the 5 classes.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling to get the semantic embedding
        feature = self.pooler(last_hidden_state, attention_mask)

        # Ensure structural_features matches the dtype and device of the transformer output
        # (Important for Mixed Precision training)
        structural_features = structural_features.to(
            dtype=feature.dtype, device=feature.device
        )

        # Fuse semantic embedding with structural features
        combined_feature = torch.cat([feature, structural_features], dim=1)

        # Pass through classification head
        output = self.fc_dropout(combined_feature)
        logits = self.fc(output)

        return logits
