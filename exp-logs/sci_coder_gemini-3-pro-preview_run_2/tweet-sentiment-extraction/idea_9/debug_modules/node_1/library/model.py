import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling:
    Computes a learnable weighted average of the last `num_hidden_layers` hidden states.
    This allows the model to leverage both high-level semantic features (top layers)
    and low-level syntactic features (lower layers) for accurate span boundary detection.
    """

    def __init__(self, num_hidden_layers=4):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights to 0.0, so softmax gives equal weight (1/N) initially.
        # These weights are learned during training.
        self.weights = nn.Parameter(torch.zeros(num_hidden_layers))

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states: Tuple of tensors (batch, seq_len, hidden_size)
                               returned by the backbone (includes embedding + all layers).
        Returns:
            weighted_output: Tensor of shape (batch, seq_len, hidden_size)
        """
        # Select the last N layers
        # all_hidden_states contains (embedding_layer, layer_1, ..., layer_N)
        selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack to shape: (num_layers, batch, seq_len, hidden_size)
        stacked_layers = torch.stack(selected_layers)

        # Compute normalized attention weights
        # shape: (num_layers)
        w = torch.softmax(self.weights, dim=0)

        # Reshape for broadcasting: (num_layers, 1, 1, 1)
        w = w.view(-1, 1, 1, 1)

        # Compute weighted sum across the layer dimension
        # shape: (batch, seq_len, hidden_size)
        weighted_output = (stacked_layers * w).sum(dim=0)

        return weighted_output


class TweetModel(nn.Module):
    """
    Tweet Sentiment Extraction Model.
    Backbone: DeBERTa-v3-Large
    Head: Weighted Layer Pooling (Last 4 layers) + Linear Projection
    """

    def __init__(self, model_path=Config.MODEL_PATH):
        super(TweetModel, self).__init__()

        # Load configuration with output_hidden_states=True to enable layer pooling
        self.config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)

        # Load the pre-trained backbone
        self.backbone = AutoModel.from_pretrained(model_path, config=self.config)

        # Custom Pooling Layer (aggregates the last 4 layers)
        self.pooling = WeightedLayerPooling(num_hidden_layers=4)

        # Prediction Head: Projects hidden_size -> 2 (start_logit, end_logit)
        self.fc = nn.Linear(self.config.hidden_size, 2)

        # Initialize the head weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Standard initialization for the linear head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) - Optional

        Returns:
            start_logits: (batch, seq_len)
            end_logits: (batch, seq_len)
        """
        # Pass inputs through the backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract hidden states
        # outputs.hidden_states is a tuple of (batch, seq_len, hidden_size)
        hidden_states = outputs.hidden_states

        # Apply Weighted Layer Pooling to combine features from the last 4 layers
        feature = self.pooling(hidden_states)

        # Project to logits
        logits = self.fc(feature)  # (batch, seq_len, 2)

        # Split logits into start and end predictions
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
