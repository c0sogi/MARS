import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Computes a weighted average of the hidden states based on a learnable attention mechanism.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [Batch, Seq_Len, Hidden]
        # attention_mask: [Batch, Seq_Len]

        # Calculate raw attention scores
        w = self.attention(last_hidden_state)  # [Batch, Seq_Len, 1]

        # Mask padding tokens so they don't contribute to the softmax
        # attention_mask is 1 for valid tokens, 0 for padding.
        # We set padding positions to a very large negative number.
        mask = attention_mask.unsqueeze(-1).float()  # [Batch, Seq_Len, 1]
        w = w.masked_fill(mask == 0, -1e9)

        # Normalize scores to probabilities
        weights = torch.softmax(w, dim=1)  # [Batch, Seq_Len, 1]

        # Weighted sum of hidden states
        # [Batch, Seq_Len, Hidden] * [Batch, Seq_Len, 1] -> Sum over Seq_Len -> [Batch, Hidden]
        feature_vector = torch.sum(last_hidden_state * weights, dim=1)

        return feature_vector


class HybridDeberta(nn.Module):
    """
    Hybrid model combining DeBERTa-v3-large embeddings with manual structural features.
    """

    def __init__(self):
        super(HybridDeberta, self).__init__()

        # Load Configuration and Pre-trained Model
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.model = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Enable Gradient Checkpointing to save memory
        self.model.gradient_checkpointing_enable()

        # Attention Pooling Layer
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Classification Head
        # Input Dimension = Semantic Embedding (Hidden Size) + Handcrafted Features
        combined_dim = self.config.hidden_size + Config.handcrafted_features_dim

        self.fc = nn.Linear(combined_dim, Config.num_classes)

        # Initialize weights for custom layers
        self._init_weights(self.fc)
        self._init_weights(self.pooler)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head layers using the backbone's init range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for submodule in module:
                self._init_weights(submodule)

    def forward(self, input_ids, attention_mask, features):
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token ids [Batch, Seq_Len]
            attention_mask: Tensor of attention masks [Batch, Seq_Len]
            features: Tensor of manual features [Batch, Feature_Dim]

        Returns:
            logits: Class logits [Batch, Num_Classes]
        """
        # 1. Get Transformer Outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Pool Hidden States to get Semantic Embedding
        embedding = self.pooler(last_hidden_state, attention_mask)

        # 3. Concatenate with Manual Features
        # Ensure features are on the same device and dtype
        features = features.to(embedding.device).type(embedding.dtype)
        combined_vector = torch.cat([embedding, features], dim=1)

        # 4. Classification
        logits = self.fc(combined_vector)

        return logits
