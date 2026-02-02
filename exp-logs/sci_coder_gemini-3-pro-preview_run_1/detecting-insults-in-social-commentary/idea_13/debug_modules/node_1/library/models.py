import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class HybridModel(nn.Module):
    """
    A Hybrid Transformer model that fuses text embeddings with structural SVD features.

    Architecture:
    1. Transformer Backbone (DeBERTa or RoBERTa)
    2. Structural Fusion: [CLS] Token + LayerNorm(SVD Features)
    3. Variable-Rate Multi-Sample Dropout (VR-MSD) Head
    """

    def __init__(self, model_name, svd_dim=Config.svd_components, pretrained=True):
        """
        Args:
            model_name (str): HuggingFace model identifier (e.g., 'microsoft/deberta-v3-large').
            svd_dim (int): Dimension of the SVD feature vector.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()

        # 1. Load Transformer Backbone
        if pretrained:
            self.config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing to save memory
        self.backbone.gradient_checkpointing_enable()

        # 2. Define Fusion Layers
        self.hidden_size = self.config.hidden_size
        self.svd_layer_norm = nn.LayerNorm(svd_dim)
        self.fusion_dim = self.hidden_size + svd_dim

        # 3. Define VR-MSD Head
        # Create a list of dropout layers with different rates defined in Config
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.dropout_rates])

        # Shared Linear Classifier
        self.fc = nn.Linear(self.fusion_dim, 1)

        # Initialize weights for custom layers
        self._init_weights(self.fc)
        self._init_weights(self.svd_layer_norm)

    def _init_weights(self, module):
        """
        Initializes weights for the custom linear and normalization layers.
        Uses the initializer range from the transformer config.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, svd_features):
        """
        Forward pass of the hybrid model.

        Args:
            input_ids (torch.Tensor): Token indices.
            attention_mask (torch.Tensor): Attention mask.
            svd_features (torch.Tensor): Dense SVD feature vector.

        Returns:
            torch.Tensor: Averaged logits from the MSD head.
        """
        # 1. Transformer Forward Pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token embedding
        # Shape: (batch_size, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # 2. Structural Fusion
        # Normalize SVD features
        svd_norm = self.svd_layer_norm(svd_features)

        # Concatenate [CLS] and SVD features
        # Shape: (batch_size, hidden_size + svd_dim)
        fused_features = torch.cat([cls_embedding, svd_norm], dim=1)

        # 3. VR-MSD Head Forward Pass
        # Apply each dropout rate, pass through the linear layer, and collect logits
        logits_list = []
        for dropout in self.dropouts:
            dropped_features = dropout(fused_features)
            logits_list.append(self.fc(dropped_features))

        # Stack logits: (num_dropouts, batch_size, 1)
        stacked_logits = torch.stack(logits_list, dim=0)

        # Average logits across the dropout samples
        # Shape: (batch_size, 1)
        avg_logits = torch.mean(stacked_logits, dim=0)

        return avg_logits
