import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling (Weighted Layer Pooling).
    Learns a weighting mask to emphasize important tokens while down-weighting padding.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (1 for token, 0 for padding)
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        # Calculate attention scores
        # w shape: [batch_size, seq_len, 1]
        w = self.attention(last_hidden_state)

        # Squeeze to [batch_size, seq_len]
        w = w.squeeze(-1)

        # Mask padding tokens: set their score to -inf so softmax becomes 0
        # attention_mask is 1 for tokens, 0 for padding
        w.masked_fill_(~attention_mask.bool(), -1e9)

        # Apply softmax to get normalized weights
        weights = torch.softmax(w, dim=1)  # [batch_size, seq_len]

        # Compute weighted sum of hidden states
        # weights.unsqueeze(-1) shape: [batch_size, seq_len, 1]
        # Broadcast multiplication and sum over sequence dimension
        embeddings = torch.sum(last_hidden_state * weights.unsqueeze(-1), dim=1)

        return embeddings


class EssayModel(nn.Module):
    """
    DeBERTa-v3-Large based model for Essay Scoring.
    Includes Gradient Checkpointing and Attention Pooling.
    """

    def __init__(self, checkpoint_path=None, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.model_name)

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing to save memory
        # This is essential for training 'Large' models on limited VRAM
        self.backbone.gradient_checkpointing_enable()

        # Pooling Mechanism
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Regression Head (Stage 1 target)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc)

        # Load state dict if checkpoint provided
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head and pooler using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Returns:
            dict: {
                "logits": Regression score [batch_size, 1],
                "embeddings": Pooled representation [batch_size, hidden_size]
            }
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling
        embeddings = self.pooler(last_hidden_state, attention_mask)

        # Predict Score
        logits = self.fc(embeddings)

        return {"logits": logits, "embeddings": embeddings}
