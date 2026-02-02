import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, logging as hf_logging
from library.config import Config

# Suppress Transformers logging to keep output clean
hf_logging.set_verbosity_error()


class StructuralAdapter(nn.Module):
    """
    Adapts the SVD structural features for fusion with Transformer embeddings.
    Applies a Dense projection followed by Layer Normalization.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        x = self.dense(x)
        x = self.norm(x)
        return x


class VariableRateMultiSampleDropout(nn.Module):
    """
    Applies multiple dropout rates to the input features, passes them through
    a shared classifier, and averages the results. This technique (VR-MSD)
    helps in regularizing the model and smoothing the decision boundary.
    """

    def __init__(self, input_dim, output_dim, dropout_rates):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.classifier = nn.Linear(input_dim, output_dim)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        logits = []
        for dropout in self.dropouts:
            # Apply dropout and then the shared classifier
            logits.append(self.classifier(dropout(x)))

        # Stack results and compute the mean across the dropout samples
        logits = torch.stack(logits, dim=0)
        return torch.mean(logits, dim=0)


class HybridDeberta(nn.Module):
    """
    Hybrid architecture combining DeBERTa-v3 backbone with SVD structural features.

    Architecture:
    1. DeBERTa-v3 [CLS] embedding.
    2. SVD features -> StructuralAdapter (Dense + LayerNorm).
    3. Concatenation of (1) and (2).
    4. Fused vector -> VariableRateMultiSampleDropout -> Logits.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)

        # Load backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                Config.MODEL_NAME, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Structural Adapter
        self.structural_adapter = StructuralAdapter(Config.SVD_COMPONENTS)

        # Determine fusion dimension
        # DeBERTa hidden size (e.g., 768) + SVD components (e.g., 256)
        self.fusion_dim = self.config.hidden_size + Config.SVD_COMPONENTS

        # Classification Head
        self.head = VariableRateMultiSampleDropout(
            input_dim=self.fusion_dim, output_dim=1, dropout_rates=Config.DROPOUT_RATES
        )

        # Enable gradient checkpointing for memory efficiency
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(self, input_ids, attention_mask, svd_features):
        """
        Forward pass of the hybrid model.

        Args:
            input_ids: Tensor of token IDs.
            attention_mask: Tensor of attention masks.
            svd_features: Tensor of structural SVD features.

        Returns:
            logits: The raw output scores (before sigmoid).
        """
        # 1. Backbone Forward Pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Extract [CLS] token embedding
        # Shape: (batch_size, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 3. Process Structural Features
        # Shape: (batch_size, svd_dim)
        struct_embedding = self.structural_adapter(svd_features)

        # 4. Feature Fusion
        # Concatenate along the feature dimension
        fused_embedding = torch.cat([cls_embedding, struct_embedding], dim=1)

        # 5. Classification Head
        logits = self.head(fused_embedding)

        return logits
