import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer that computes a weighted average of hidden states
    based on a learned attention score for each token.
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
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        # Calculate attention weights
        w = self.attention(last_hidden_state)  # [batch_size, seq_len, 1]

        # Mask padding tokens
        float_mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        w = w.masked_fill(float_mask == 0, -1e9)

        # Normalize weights
        w = torch.softmax(w, dim=1)  # [batch_size, seq_len, 1]

        # Weighted sum
        feature = torch.sum(w * last_hidden_state, dim=1)  # [batch_size, hidden_size]
        return feature


class CustomModel(nn.Module):
    """
    Main model architecture using DeBERTa-v3-large backbone, Attention Pooling,
    Multi-Sample Dropout, and a Hybrid Head structure (Regression + Classification).
    """

    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = Config

        # Determine model name/path (can be a HF hub name or local path to DAPT model)
        model_name = config_path if config_path else self.cfg.model_name

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        # Initialize Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Pooling Layer
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout (MSD)
        # Using 5 dropout layers to average gradients and improve generalization
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Heads
        # 1. Regression Head (Continuous Score)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.num_classes)

        # 2. Classification Head (Discrete Buckets for Hybrid Loss)
        self.fc_class = nn.Linear(self.config.hidden_size, self.cfg.hybrid_num_classes)

        # Initialize weights for custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc)
        self._init_weights(self.fc_class)

    def _init_weights(self, module):
        """
        Custom weight initialization for added layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for sub_module in module:
                self._init_weights(sub_module)

    def feature(self, input_ids, attention_mask):
        """
        Extracts pooled features from the backbone.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pooler(last_hidden_state, attention_mask)
        return feature

    def forward(self, input_ids, attention_mask):
        """
        Forward pass with Multi-Sample Dropout applied to both heads.
        """
        feature = self.feature(input_ids, attention_mask)

        # Apply MSD for Regression Head
        logits_sum = 0
        for dropout in self.dropouts:
            logits_sum += self.fc(dropout(feature))
        logits = logits_sum / len(self.dropouts)

        # Apply MSD for Classification Head
        class_logits_sum = 0
        for dropout in self.dropouts:
            class_logits_sum += self.fc_class(dropout(feature))
        class_logits = class_logits_sum / len(self.dropouts)

        return {"logits": logits, "class_logits": class_logits}
