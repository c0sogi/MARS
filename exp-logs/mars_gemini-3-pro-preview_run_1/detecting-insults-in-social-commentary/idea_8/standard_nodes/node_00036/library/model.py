import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class VariableRateMSD(nn.Module):
    """
    Variable-Rate Multi-Sample Dropout (VR-MSD).
    Applies multiple dropout masks with different rates to the input features,
    passes them through a shared linear layer, and averages the results.
    This technique acts as an internal ensemble, improving robustness and generalization.
    """

    def __init__(self, input_dim, output_dim, dropout_rates):
        """
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output (number of classes).
            dropout_rates (list of float): List of dropout probabilities to apply.
        """
        super().__init__()
        self.dropout_rates = dropout_rates
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Averaged output logits of shape (batch_size, output_dim).
        """
        # Apply each dropout mask, pass through the shared FC layer, and sum outputs
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.fc(dropout(x))
            else:
                output += self.fc(dropout(x))

        # Return the average
        return output / len(self.dropout_rates)


class HybridDebertaModel(nn.Module):
    """
    Hybrid DeBERTa-v3 with Normalized Structural Fusion.
    Combines semantic embeddings from DeBERTa-v3 with statistical structural features (SVD).
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load pre-trained weights for the backbone.
        """
        super().__init__()
        self.config = Config

        # 1. Semantic Backbone (DeBERTa-v3)
        # We load the config first to access hidden_size
        model_config = AutoConfig.from_pretrained(self.config.model_name)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                self.config.model_name, config=model_config
            )
        else:
            self.backbone = AutoModel.from_config(model_config)

        # 2. Structural Branch Normalization
        # LayerNorm is critical here to align the scale of SVD features (which are standard scaled)
        # with the transformer embeddings (which have their own internal normalization).
        self.struct_norm = nn.LayerNorm(self.config.svd_output_dim)

        # 3. Fusion Dimension
        # Concatenation of [CLS] embedding and Structural features
        self.fusion_dim = model_config.hidden_size + self.config.svd_output_dim

        # 4. Classification Head (VR-MSD)
        self.head = VariableRateMSD(
            input_dim=self.fusion_dim,
            output_dim=1,  # Binary classification
            dropout_rates=self.config.dropout_rates,
        )

        # Initialize weights for the custom head
        self._init_weights(self.head.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.backbone.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, struct_features):
        """
        Forward pass of the hybrid model.

        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer.
            attention_mask (torch.Tensor): Attention mask from tokenizer.
            struct_features (torch.Tensor): Dense SVD features of shape (batch, svd_dim).

        Returns:
            torch.Tensor: Logits of shape (batch, 1).
        """
        # 1. Extract Semantic Features
        # We use the [CLS] token (index 0) from the last hidden state
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[
            :, 0, :
        ]  # Shape: [batch_size, hidden_size]

        # 2. Process Structural Features
        # Apply LayerNorm to the structural input
        norm_struct = self.struct_norm(struct_features)  # Shape: [batch_size, svd_dim]

        # 3. Feature Fusion
        # Concatenate along the feature dimension
        fused_features = torch.cat(
            [cls_embedding, norm_struct], dim=1
        )  # Shape: [batch_size, fusion_dim]

        # 4. Classification
        logits = self.head(fused_features)

        return logits
