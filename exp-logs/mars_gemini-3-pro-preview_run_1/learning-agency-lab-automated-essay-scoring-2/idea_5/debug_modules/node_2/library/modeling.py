import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling (also known as Weighted Layer Pooling in some contexts,
    though strictly refers here to weighting tokens/sentences).
    Applies an attention mechanism to the last hidden state to dynamically
    weight the importance of different tokens/sentences in the essay.
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
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # Calculate attention weights: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        if attention_mask is not None:
            # Mask padding tokens so they don't contribute to the average
            # attention_mask is 1 for tokens, 0 for padding.
            # We want to set padding positions to -inf before softmax.
            padding_mask = attention_mask.unsqueeze(-1) == 0
            w = w.masked_fill(padding_mask, -1e4)

        # Softmax over the sequence dimension
        w = torch.softmax(w, dim=1)

        # Weighted sum of hidden states: (batch_size, hidden_size)
        pooled_output = torch.sum(w * last_hidden_state, dim=1)
        return pooled_output


class EssayScorer(nn.Module):
    """
    Essay Scoring Model wrapping DeBERTa-v3-Large with Attention Pooling
    and a Regression Head.
    """

    def __init__(self, model_name_or_path=None, pretrained=True):
        super(EssayScorer, self).__init__()

        # Use Config model name if not provided
        if model_name_or_path is None:
            model_name_or_path = Config.model_name

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name_or_path)

        # Apply Dropout Settings from Config (0.0 for regression stability)
        self.config.hidden_dropout_prob = Config.hidden_dropout_prob
        self.config.attention_probs_dropout_prob = Config.attention_probs_dropout_prob
        self.config.use_cache = False

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name_or_path, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing
        # Critical for Large models with max_len=1024 on limited VRAM
        self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pool = AttentionPooling(self.config.hidden_size)

        # Regression Head
        self.fc = nn.Linear(self.config.hidden_size, Config.num_labels)

        # Initialize weights for custom layers
        self._init_weights(self.pool)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom pooling and head layers.
        Uses the initializer range from the backbone config.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, input_ids, attention_mask):
        """
        Extract features from the backbone and pooling layer.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, attention_mask)
        return feature

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        Returns logits (scalar scores).
        """
        # Get pooled representation
        feature = self.feature(input_ids, attention_mask)

        # Regression Output
        logits = self.fc(feature)

        # Squeeze to shape (batch_size,)
        return logits.squeeze(-1)
