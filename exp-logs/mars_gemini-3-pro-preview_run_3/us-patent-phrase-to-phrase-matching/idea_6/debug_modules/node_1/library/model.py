import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import cfg


class AttentionPooling(nn.Module):
    """
    Attention Pooling mechanism to aggregate token embeddings.
    Computes a weighted sum of hidden states based on a learned query.
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
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        # Compute attention scores
        w = self.attention(last_hidden_state)  # [batch_size, seq_len, 1]

        # Mask padding tokens (set to -inf so softmax becomes 0)
        float_mask = attention_mask.unsqueeze(-1).float()
        w = w.masked_fill(float_mask == 0, -1e4)

        # Compute weights
        weights = torch.softmax(w, dim=1)

        # Weighted sum of hidden states
        pooled_output = torch.sum(weights * last_hidden_state, dim=1)
        return pooled_output


class CustomModel(nn.Module):
    """
    Main model class for Phrase Matching.
    Uses DeBERTa-v3-large backbone with Attention Pooling and Dual Heads (Regression + Classification).
    """

    def __init__(self, pretrained_path=None):
        super().__init__()

        # Determine model path:
        # 1. Use explicit pretrained_path if provided (e.g., for inference)
        # 2. Use DAPT model if available in working directory
        # 3. Fallback to base model name from config
        if pretrained_path:
            self.model_path = pretrained_path
        elif os.path.exists(cfg.dapt_model_dir) and (
            os.path.exists(os.path.join(cfg.dapt_model_dir, "config.json"))
            or os.path.exists(os.path.join(cfg.dapt_model_dir, "model.safetensors"))
            or os.path.exists(os.path.join(cfg.dapt_model_dir, "pytorch_model.bin"))
        ):
            self.model_path = cfg.dapt_model_dir
        else:
            self.model_path = cfg.model_name

        # Load Configuration
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.config.output_hidden_states = True

        # Load Backbone
        self.backbone = AutoModel.from_pretrained(self.model_path, config=self.config)

        # Enable Gradient Checkpointing for memory efficiency
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Dropout
        self.dropout = nn.Dropout(cfg.dropout)

        # Heads
        # Regression Head (Output: scalar score)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Classification Head (Output: logits for 5 classes)
        self.fc_class = nn.Linear(self.config.hidden_size, cfg.num_classes)

        # Initialize weights for custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc)
        self._init_weights(self.fc_class)

    def _init_weights(self, module):
        """
        Initialize weights for the custom heads and pooling layers.
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

    def feature(self, input_ids, attention_mask):
        """
        Extract features from the backbone and pooling layer.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pooler(last_hidden_state, attention_mask)
        return feature

    def forward(self, input_ids, attention_mask, labels=None, class_labels=None):
        """
        Forward pass.
        Returns a dictionary containing logits for both regression and classification heads.
        """
        feature = self.feature(input_ids, attention_mask)
        output_feature = self.dropout(feature)

        logits = self.fc(output_feature)
        class_logits = self.fc_class(output_feature)

        return {"logits": logits, "class_logits": class_logits}
