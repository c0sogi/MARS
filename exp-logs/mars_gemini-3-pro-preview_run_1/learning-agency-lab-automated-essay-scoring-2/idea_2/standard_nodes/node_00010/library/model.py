import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class EssayRegressor(nn.Module):
    """
    Neural network model for essay scoring using DeBERTa-v3-small backbone.
    Implements Contextual Mean Pooling and a linear regression head.
    """

    def __init__(self, model_name=Config.model_name, pretrained=True):
        """
        Initializes the EssayRegressor model.

        Args:
            model_name (str): Name of the pre-trained model to load.
            pretrained (bool): Whether to load pre-trained weights.
        """
        super(EssayRegressor, self).__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Load backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Regression Head: Maps hidden size to a single scalar score
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the regression head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the specific module using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature_pooling(self, last_hidden_states, attention_mask):
        """
        Performs Contextual Mean Pooling on the last hidden states.

        Args:
            last_hidden_states (torch.Tensor): Output from the backbone (batch, seq_len, hidden_size).
            attention_mask (torch.Tensor): Attention mask (batch, seq_len).

        Returns:
            torch.Tensor: Pooled embeddings (batch, hidden_size).
        """
        # Expand attention_mask to match hidden_states dimensions: (batch, seq_len, 1)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        )

        # Sum the embeddings of valid tokens (ignoring padding)
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)

        # Count the number of valid tokens (avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Compute the mean
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Predicted scores (batch, 1).
        """
        # Pass inputs through the backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the sequence of hidden states from the last layer
        last_hidden_state = outputs.last_hidden_state

        # Apply Contextual Mean Pooling
        pooled_output = self.feature_pooling(last_hidden_state, attention_mask)

        # Pass through the regression head
        logits = self.fc(pooled_output)

        return logits
