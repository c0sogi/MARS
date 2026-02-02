import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class InsultDetector(nn.Module):
    """
    Hybrid DeBERTa-v3 model with Multi-Sample Dropout and Structural Feature Injection.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": Config.dropout,
                "attention_probs_dropout_prob": Config.dropout,
                "num_labels": Config.num_classes,
            }
        )

        # 2. Semantic Backbone
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # 3. Structural Branch (SVD Projection)
        # Projects SVD dim (256) -> Hidden dim (768)
        self.svd_projection = nn.Sequential(
            nn.Linear(Config.svd_components, Config.hidden_size),
            nn.LayerNorm(Config.hidden_size),
            nn.Dropout(Config.dropout),
        )

        # 4. Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.msd_dropout) for _ in range(Config.msd_num)]
        )

        # 5. Classifier
        # Input dim is hidden_size * 2 because we concatenate [CLS] and SVD
        self.classifier = nn.Linear(Config.hidden_size * 2, Config.num_classes)

        # Initialize weights for custom layers
        self._init_custom_weights()

    def _init_custom_weights(self):
        """Initialize weights for the new layers."""
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

        for module in self.svd_projection:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, svd_features):
        """
        Forward pass of the hybrid model.

        Args:
            input_ids: Tensor (batch, seq_len)
            attention_mask: Tensor (batch, seq_len)
            svd_features: Tensor (batch, svd_components)

        Returns:
            logits: Tensor (batch, 1)
        """
        # 1. Get Backbone Outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation from the last hidden state
        cls_token = outputs.last_hidden_state[:, 0, :]

        # 2. Process Structural Features
        svd_emb = self.svd_projection(svd_features)

        # 3. Concatenation
        features = torch.cat([cls_token, svd_emb], dim=1)

        # 4. Multi-Sample Dropout & Classification
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.classifier(dropout(features)))

        # Average logits
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits
