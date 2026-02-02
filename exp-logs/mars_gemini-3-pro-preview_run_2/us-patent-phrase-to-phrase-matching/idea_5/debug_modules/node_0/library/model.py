import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class AttentionPooling(nn.Module):
    """
    Applies attention mechanism to the last hidden state of the transformer
    to generate a fixed-size semantic embedding.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Calculate attention weights: [batch_size, seq_len, 1]
        w = self.attention(last_hidden_state)

        # Mask padding tokens so they don't contribute to the average
        mask = attention_mask.unsqueeze(-1).float()
        w = w.masked_fill(mask == 0, -1e9)

        # Normalize weights
        w = torch.softmax(w, dim=1)

        # Weighted sum: [batch_size, hidden_size]
        pooled_output = torch.sum(w * last_hidden_state, dim=1)
        return pooled_output


class DebertaV3FeatureFused(nn.Module):
    """
    Hybrid architecture combining DeBERTa-v3-large semantic embeddings with
    explicit structural features (Levenshtein, Jaccard, Length Ratio).
    """

    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_classes=5,
        num_structural_features=3,
        pretrained=True,
    ):
        super().__init__()

        # Load Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Semantic Stream: Attention Pooling
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Fusion & Classification Head
        # Input dimension = Transformer Hidden Size + Number of Structural Features
        fusion_dim = self.config.hidden_size + num_structural_features
        self.fc = nn.Linear(fusion_dim, num_classes)

        # Initialize weights for the custom head
        self._init_weights(self.pooler)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize weights for the custom layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, structural_features):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            structural_features: [batch_size, num_structural_features]

        Returns:
            logits: [batch_size, num_classes]
        """
        # 1. Semantic Stream
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        semantic_embedding = self.pooler(last_hidden_state, attention_mask)

        # 2. Fusion
        # Concatenate dense semantic embedding with explicit structural features
        combined_features = torch.cat([semantic_embedding, structural_features], dim=1)

        # 3. Classification
        logits = self.fc(combined_features)

        return logits
