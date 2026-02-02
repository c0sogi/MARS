import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class EssayDebertaModel(nn.Module):
    """
    Neural network model for essay scoring based on DeBERTa-v3-Large.

    Architecture:
    1. Backbone: DeBERTa-v3-Large
    2. Pooling: Concatenation of Mean Pooling and Max Pooling
    3. Head: Linear Regression (Scalar Output)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)

        # Apply dropout settings from Config
        self.config.attention_probs_dropout_prob = Config.dropout
        self.config.hidden_dropout_prob = Config.dropout

        # Load Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.model = AutoModel.from_config(self.config)

        # Regression Head
        # Input size is doubled because we concatenate Mean and Max pooling
        self.fc = nn.Linear(self.config.hidden_size * 2, 1)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the linear head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature_pooling(self, last_hidden_state, attention_mask):
        """
        Applies Concatenated Mean and Max Pooling.

        Args:
            last_hidden_state (torch.Tensor): Output from backbone [Batch, Seq_Len, Hidden]
            attention_mask (torch.Tensor): Attention mask [Batch, Seq_Len]

        Returns:
            torch.Tensor: Pooled embeddings [Batch, Hidden * 2]
        """
        # Expand mask to match embedding dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # --- Mean Pooling ---
        # Sum embeddings ignoring padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # Count non-padding tokens (clamp to avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # --- Max Pooling ---
        # Create a copy of embeddings to mask padding
        embeddings = last_hidden_state.clone()
        # Set padding tokens to a very small number so they are not selected by max
        embeddings[input_mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(embeddings, 1)

        # --- Concatenate ---
        pooled_output = torch.cat([mean_embeddings, max_embeddings], 1)

        return pooled_output

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Scalar score prediction [Batch, 1]
        """
        # Get backbone outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply custom pooling
        feature = self.feature_pooling(last_hidden_state, attention_mask)

        # Project to scalar score
        output = self.fc(feature)

        return output
