import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class HybridDeberta(nn.Module):
    """
    Hybrid DeBERTa-v3 model that fuses text embeddings with structural features (SVD).
    Implements Variable-Rate Multi-Sample Dropout (VR-MSD) for robust regularization.
    """

    def __init__(self, pretrained_model_name_or_path=None):
        super(HybridDeberta, self).__init__()

        # Use Config model name if no specific path is provided
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = Config.MODEL_NAME

        # Load configuration
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Load Backbone
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=self.config
        )

        # Structural Feature Branch: Normalization
        # Normalize the dense SVD vector before fusion
        self.svd_layer_norm = nn.LayerNorm(Config.SVD_COMPONENTS)

        # Calculate Fusion Dimension
        # DeBERTa [CLS] dim + SVD dim
        self.fusion_dim = self.config.hidden_size + Config.SVD_COMPONENTS

        # Variable-Rate Multi-Sample Dropout (VR-MSD)
        # Create parallel dropout layers with different rates defined in Config
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.DROPOUT_RATES])

        # Shared Classification Head
        self.fc = nn.Linear(self.fusion_dim, 1)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head and layer norm.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, svd_features, label=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer.
            attention_mask (torch.Tensor): Attention mask from tokenizer.
            svd_features (torch.Tensor): Dense structural features (SVD).
            label (torch.Tensor, optional): Labels (not used directly in forward, but kept for signature compatibility).

        Returns:
            torch.Tensor: Averaged logits from the multi-sample dropout heads.
        """
        # 1. Backbone Extraction
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token (index 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 2. Structural Branch Processing
        svd_norm = self.svd_layer_norm(svd_features)

        # 3. Fusion
        fused_features = torch.cat([cls_embedding, svd_norm], dim=1)

        # 4. Variable-Rate Multi-Sample Dropout & Classification
        logits_list = []
        for dropout in self.dropouts:
            # Apply specific dropout rate
            dropped_features = dropout(fused_features)
            # Pass through shared linear layer
            logits_list.append(self.fc(dropped_features))

        # Stack logits: (num_dropouts, batch_size, 1)
        stacked_logits = torch.stack(logits_list, dim=0)

        # Average logits across all dropout samples
        avg_logits = torch.mean(stacked_logits, dim=0)

        # Remove last dimension to return (batch_size,)
        return avg_logits.squeeze(-1)
