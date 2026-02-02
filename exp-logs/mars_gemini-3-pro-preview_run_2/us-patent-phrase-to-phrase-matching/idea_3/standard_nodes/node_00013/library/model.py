import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class CustomDeberta(nn.Module):
    """
    Custom DeBERTa-v3-large architecture for Phrase Similarity.

    Features:
    - Backbone: microsoft/deberta-v3-large
    - Multi-Layer Fusion: Concatenates [CLS] tokens from the last 4 layers.
    - Weighted Projection: Fuses the concatenated features via a Linear layer.
    - Head: 5-class classification output.
    """

    def __init__(self, model_name=Config.model_name, pretrained=True):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True

        # Initialize Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # --- Multi-Layer Fusion Setup ---
        self.num_layers_fusion = 4
        self.hidden_size = self.config.hidden_size

        # Fusion Projection Layer
        # Input: Concatenation of 4 layers (4 * hidden_size)
        # Output: Projected back to hidden_size
        self.fusion_layer = nn.Linear(
            self.hidden_size * self.num_layers_fusion, self.hidden_size
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.Tanh()

        # --- Classification Head ---
        self.dropout = nn.Dropout(Config.fc_dropout)
        self.fc = nn.Linear(self.hidden_size, Config.target_size)

        # Initialize weights for custom layers
        self._init_weights(self.fusion_layer)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initializes weights for the custom linear layers using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits of shape (batch_size, 5).
        """
        # Pass through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Retrieve all hidden states (tuple of tensors)
        # outputs.hidden_states contains: (embeddings, layer_1, ..., layer_24)
        all_hidden_states = outputs.hidden_states

        # Extract [CLS] tokens (index 0) from the last 'num_layers_fusion' layers
        cls_embeddings = []
        for i in range(1, self.num_layers_fusion + 1):
            # Access layers from the end: -1, -2, -3, -4
            layer_output = all_hidden_states[-i]
            # Take the [CLS] token embedding
            cls_token = layer_output[:, 0, :]
            cls_embeddings.append(cls_token)

        # Concatenate features along the feature dimension
        # Shape: (batch_size, 4 * hidden_size)
        fused_features = torch.cat(cls_embeddings, dim=1)

        # Apply Weighted Projection (Fusion)
        x = self.fusion_layer(fused_features)
        x = self.layer_norm(x)
        x = self.activation(x)

        # Apply Classification Head
        x = self.dropout(x)
        logits = self.fc(x)

        return logits
