import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.configuration import Config


class HybridModel(nn.Module):
    """
    Dual-Stream Hybrid Architecture for Insult Detection.
    Integrates a Transformer backbone with SVD-based structural features via Normalized SVD Fusion.
    Implements Variable-Rate Multi-Sample Dropout (VR-MSD) for robust regularization.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        """
        Args:
            model_name (str): HuggingFace model identifier (e.g., 'microsoft/deberta-v3-large').
            pretrained (bool): Whether to load pre-trained weights.
        """
        super(HybridModel, self).__init__()
        self.config = Config

        # 1. Load Transformer Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        # Enable Gradient Checkpointing to save memory
        # self.backbone.gradient_checkpointing_enable()

        # Dynamically retrieve hidden size (e.g., 1024 for Large, 768 for Base)
        self.hidden_size = self.backbone.config.hidden_size

        # 2. Structural Fusion Components
        # Layer Normalization for SVD features before concatenation
        self.svd_layer_norm = nn.LayerNorm(self.config.svd_embedding_size)

        # Calculate fused dimension: Transformer [CLS] + SVD Features
        self.fused_dim = self.hidden_size + self.config.svd_embedding_size

        # 3. Variable-Rate Multi-Sample Dropout (VR-MSD) Head
        # We create multiple dropout layers with different probabilities
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in self.config.dropout_rates]
        )

        # Single Linear Layer shared across all dropout samples
        self.fc = nn.Linear(self.fused_dim, 1)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            # Use the initializer range from the backbone config for consistency
            module.weight.data.normal_(
                mean=0.0, std=self.backbone.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, svd_features):
        """
        Forward pass of the hybrid model.

        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer. Shape: (batch_size, max_len)
            attention_mask (torch.Tensor): Attention mask. Shape: (batch_size, max_len)
            svd_features (torch.Tensor): Structured features. Shape: (batch_size, svd_embedding_size)

        Returns:
            torch.Tensor: Averaged logits. Shape: (batch_size, 1)
        """
        # 1. Backbone Extraction
        # We use the last_hidden_state.
        # For DeBERTa and RoBERTa, index 0 is the [CLS] or start token equivalent.
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[
            :, 0, :
        ]  # Shape: (batch_size, hidden_size)

        # 2. Normalized SVD Fusion
        # Normalize SVD features to match the scale of embeddings
        svd_norm = self.svd_layer_norm(svd_features)

        # Concatenate along the feature dimension
        fused_features = torch.cat(
            [cls_embedding, svd_norm], dim=1
        )  # Shape: (batch_size, fused_dim)

        # 3. VR-MSD Inference
        # Pass the fused vector through multiple dropout masks and average the predictions
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout
            x = dropout(fused_features)
            # Pass through linear layer
            logits_list.append(self.fc(x))

        # Stack logits: (num_dropouts, batch_size, 1)
        logits_stack = torch.stack(logits_list, dim=0)

        # Average logits across the dropout samples
        mean_logits = torch.mean(logits_stack, dim=0)

        return mean_logits
