import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class HybridDebertaModel(nn.Module):
    """
    Hybrid DeBERTa-v3 model with Normalized Structural Fusion and
    Variable-Rate Multi-Sample Dropout (VR-MSD).

    This model implements the architecture described in Idea 9:
    1. Semantic Backbone: Microsoft DeBERTa-v3-base extracting the [CLS] token.
    2. Structural Branch: TruncatedSVD features normalized via LayerNorm.
    3. Fusion: Concatenation of semantic and structural vectors.
    4. Classification Head: VR-MSD (parallel dropouts) -> Linear Layer -> Average.
    """

    def __init__(self, pretrained=True):
        """
        Initializes the model components.

        Args:
            pretrained (bool): Whether to load pre-trained weights for the backbone.
        """
        super(HybridDebertaModel, self).__init__()

        # Configuration
        self.model_name = Config.model_name
        self.hidden_size = Config.hidden_size
        self.svd_dim = Config.svd_dim
        self.dropout_rates = Config.dropout_rates

        # 1. Semantic Backbone
        config = AutoConfig.from_pretrained(self.model_name)
        if pretrained:
            self.backbone = AutoModel.from_pretrained(self.model_name, config=config)
        else:
            self.backbone = AutoModel.from_config(config)

        # 2. Structural Branch Normalization
        # Applies Layer Normalization to SVD features to match embedding scale
        self.svd_norm = nn.LayerNorm(self.svd_dim)

        # 3. Variable-Rate Multi-Sample Dropout (VR-MSD)
        # A list of parallel dropout layers with increasing rates
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_rates])

        # 4. Classification Head
        # Input: Concatenated [CLS] embedding + Normalized SVD features
        fusion_dim = self.hidden_size + self.svd_dim
        self.fc = nn.Linear(fusion_dim, 1)

        # Initialize weights for custom layers
        self._init_weights(self.fc)
        self._init_weights(self.svd_norm)

    def _init_weights(self, module):
        """
        Initialize weights for the new layers (Head and LayerNorm).
        The backbone weights are handled by transformers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, svd_features, target=None):
        """
        Forward pass of the hybrid model.

        Args:
            input_ids (torch.Tensor): Token indices. Shape: (batch_size, max_len)
            attention_mask (torch.Tensor): Attention mask. Shape: (batch_size, max_len)
            svd_features (torch.Tensor): Dense structural features. Shape: (batch_size, svd_dim)
            target (torch.Tensor, optional): Labels. Not used in forward computation.

        Returns:
            torch.Tensor: Logits. Shape: (batch_size, 1)
        """
        # 1. Backbone Feature Extraction
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token (index 0 of last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 2. Structural Feature Normalization
        svd_norm = self.svd_norm(svd_features)

        # 3. Feature Fusion
        fused_features = torch.cat((cls_embedding, svd_norm), dim=1)

        # 4. VR-MSD and Classification
        # Pass fused vector through multiple dropout masks and the same linear layer
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout -> Linear Classifier
            output = self.fc(dropout(fused_features))
            logits_list.append(output)

        # Average the predictions from all dropout samples
        logits = torch.mean(torch.stack(logits_list), dim=0)

        return logits
