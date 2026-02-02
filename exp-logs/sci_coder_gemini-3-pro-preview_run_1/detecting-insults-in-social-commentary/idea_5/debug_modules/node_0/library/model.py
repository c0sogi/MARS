import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a weighted average of the [CLS] tokens from the last N hidden layers.
    The weights are learnable parameters normalized via Softmax.
    """

    def __init__(self, num_pool_layers: int = 4, hidden_size: int = 768):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        self.hidden_size = hidden_size
        # Initialize weights to 0, resulting in equal weighting after Softmax initially
        self.weights = nn.Parameter(torch.zeros(num_pool_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (embeddings + layer outputs)
        # We take the last 'num_pool_layers'
        selected_layers = all_hidden_states[-self.num_pool_layers :]

        # Stack CLS tokens: (batch_size, num_pool_layers, hidden_size)
        # Assuming [CLS] is at index 0
        cls_outputs = torch.stack([layer[:, 0, :] for layer in selected_layers], dim=1)

        # Compute normalized weights: (num_pool_layers)
        norm_weights = torch.softmax(self.weights, dim=0)

        # Weighted sum: (batch_size, hidden_size)
        # Broadcasting: (batch, layers, hidden) * (1, layers, 1) -> sum over layers
        weighted_output = torch.sum(cls_outputs * norm_weights.view(1, -1, 1), dim=1)

        return weighted_output


class InsultDetector(nn.Module):
    """
    Hybrid DeBERTa-v3 model with Weighted Layer Pooling and Gated Structural Injection.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": Config.dropout,
                "attention_probs_dropout_prob": Config.dropout,
                "num_labels": Config.num_classes,
            }
        )

        # 2. Semantic Backbone
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # 3. Weighted Layer Pooling
        self.pooling = WeightedLayerPooling(
            num_pool_layers=Config.pool_layers, hidden_size=Config.hidden_size
        )

        # 4. Structural Branch (SVD Projection)
        # Projects SVD dim (256) -> Hidden dim (768)
        self.svd_projection = nn.Sequential(
            nn.Linear(Config.svd_components, Config.hidden_size),
            nn.LayerNorm(Config.hidden_size),
            nn.Dropout(Config.dropout),
        )

        # 5. Gated Fusion Mechanism
        # Gate takes concatenated [Trans; SVD] -> produces Gate vector
        self.gate_dense = nn.Linear(Config.hidden_size * 2, Config.hidden_size)

        # 6. Classifier
        self.classifier = nn.Linear(Config.hidden_size, Config.num_classes)

        # Initialize weights for custom layers
        self._init_custom_weights()

    def _init_custom_weights(self):
        """Initialize weights for the new layers (Projection, Gate, Classifier)."""
        for module in [self.gate_dense, self.classifier]:
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

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

        # 2. Weighted Pooling of Hidden States
        # outputs.hidden_states is a tuple of (batch, seq_len, hidden)
        trans_emb = self.pooling(outputs.hidden_states)

        # 3. Process Structural Features
        svd_emb = self.svd_projection(svd_features)

        # 4. Gated Fusion
        # Calculate Gate: g = sigmoid(W * [V_trans; V_svd] + b)
        combined = torch.cat([trans_emb, svd_emb], dim=1)
        gate = torch.sigmoid(self.gate_dense(combined))

        # Apply Fusion: V_final = V_trans + (g * V_svd)
        fused_emb = trans_emb + (gate * svd_emb)

        # 5. Classification
        logits = self.classifier(fused_emb)

        return logits
