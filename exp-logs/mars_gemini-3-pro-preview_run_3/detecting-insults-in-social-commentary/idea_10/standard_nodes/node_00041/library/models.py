import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomModel(nn.Module):
    """
    Custom Model for Insult Detection using HuggingFace Transformers.

    This class wraps a transformer backbone (e.g., RoBERTa, DeBERTa), applies
    layer freezing to the lower layers to preserve general linguistic features,
    uses Mean Pooling to aggregate token embeddings, and projects the result
    to a scalar logit for binary classification.
    """

    def __init__(self, model_name: str, config: Config):
        """
        Initialize the CustomModel.

        Args:
            model_name (str): The name or path of the pre-trained transformer model.
            config (Config): The configuration object containing hyperparameters
                             like dropout rate and number of layers to freeze.
        """
        super().__init__()
        self.config = config

        # Load the configuration for the backbone to access hidden_size
        self.model_config = AutoConfig.from_pretrained(model_name)

        # Load the pre-trained backbone model
        self.model = AutoModel.from_pretrained(model_name)

        # Apply layer freezing strategy
        self._freeze_layers()

        # Define the classification head
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.model_config.hidden_size, 1)

        # Initialize weights for the classification head
        self._init_weights(self.fc)

    def _freeze_layers(self):
        """
        Freezes the embeddings and the bottom N encoder layers based on the config.
        """
        # 1. Freeze Embeddings
        if hasattr(self.model, "embeddings"):
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

        # 2. Freeze Encoder Layers
        # Both RoBERTa and DeBERTa typically store layers in model.encoder.layer
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            layers = self.model.encoder.layer
            num_layers = len(layers)

            # Determine how many layers to freeze, ensuring we don't exceed available layers
            layers_to_freeze = min(self.config.freeze_layers, num_layers)

            for i in range(layers_to_freeze):
                for param in layers[i].parameters():
                    param.requires_grad = False

    def _init_weights(self, module):
        """
        Initialize the weights of the linear head using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        """
        Extracts features from the backbone using Mean Pooling.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len].

        Returns:
            torch.Tensor: Pooled embeddings [batch_size, hidden_size].
        """
        # Forward pass through the backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Perform Mean Pooling
        # Expand mask to match hidden state dimensions: [batch, seq] -> [batch, seq, 1]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings over the sequence dimension, masking out padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum the mask to get the count of valid tokens
        sum_mask = input_mask_expanded.sum(1)

        # Clamp to avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (not used by all models, but accepted).

        Returns:
            torch.Tensor: Logits of shape [batch_size].
        """
        # Extract features (ignoring token_type_ids for backbone simplicity/compatibility)
        feature = self.feature(input_ids, attention_mask)

        # Apply Dropout
        x = self.dropout(feature)

        # Apply Linear Head
        logits = self.fc(x)

        # Squeeze the last dimension to return [batch_size]
        return logits.squeeze(-1)
