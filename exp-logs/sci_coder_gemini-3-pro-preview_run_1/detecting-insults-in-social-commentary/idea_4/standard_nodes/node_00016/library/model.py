import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class HybridDeberta(nn.Module):
    """
    Robust Hybrid DeBERTa-v3 model with Multi-Sample Dropout.
    Fuses semantic features (DeBERTa [CLS]) with structural features (SVD of TF-IDF).
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)

        # Initialize Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Structural Feature Normalization
        # Normalizing SVD features ensures they don't dominate or vanish when fused
        # with the transformer embeddings.
        self.svd_layer_norm = nn.LayerNorm(Config.svd_components)

        # Calculate Fusion Dimension
        # DeBERTa-base hidden size (768) + SVD components (256)
        self.fusion_dim = Config.hidden_size + Config.svd_components

        # Multi-Sample Dropout
        # We use a list of dropout layers with different rates/seeds.
        # This technique acts like an ensemble, reducing overfitting on small datasets.
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.dropout_rates])

        # Shared Classification Head
        # Maps the fused vector to a single logit
        self.fc = nn.Linear(self.fusion_dim, 1)

        # Initialize weights for new layers
        self._init_weights(self.fc)
        self._init_weights(self.svd_layer_norm)

    def _init_weights(self, module):
        """
        Initialize weights for the new linear and normalization layers
        using the same standard deviation as the transformer backbone.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, svd_feat):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer. Shape: (Batch, Seq_Len)
            attention_mask (torch.Tensor): Attention mask. Shape: (Batch, Seq_Len)
            svd_feat (torch.Tensor): Structural SVD features. Shape: (Batch, SVD_Dim)

        Returns:
            torch.Tensor: Logits. Shape: (Batch,)
        """
        # 1. Extract Semantic Features
        # Pass through DeBERTa backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token embedding (index 0)
        # Shape: (Batch, 768)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 2. Process Structural Features
        # Normalize the SVD vector
        # Shape: (Batch, 256)
        svd_normalized = self.svd_layer_norm(svd_feat)

        # 3. Feature Fusion
        # Concatenate along the feature dimension
        # Shape: (Batch, 1024)
        fused_embedding = torch.cat((cls_embedding, svd_normalized), dim=1)

        # 4. Multi-Sample Dropout & Classification
        # Pass the fused embedding through multiple dropout masks and the shared classifier
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout -> Linear
            output = self.fc(dropout(fused_embedding))
            logits_list.append(output)

        # Stack logits and calculate the mean across the dropout samples
        # logits_list contains tensors of shape (Batch, 1)
        # stack -> (Batch, 1, num_dropouts) -> mean(dim=2) -> (Batch, 1)
        logits = torch.stack(logits_list, dim=2).mean(dim=2)

        # Squeeze to return shape (Batch,)
        return logits.squeeze(-1)
