import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomDeberta(nn.Module):
    """
    Branch A: Syntactic-Semantic Transformer
    Backbone: microsoft/deberta-v3-large
    Head: Weighted Layer Pooling (Last 4 Layers) -> Linear
    """

    def __init__(self, model_name=Config.MODEL_DEBERTA, num_classes=3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        # Learnable weights for the last 4 layers
        # Initialize to zeros so softmax yields equal weights (0.25) initially
        self.layer_weights = nn.Parameter(torch.zeros(4))

        self.fc = nn.Linear(self.config.hidden_size, num_classes)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass with Weighted Layer Pooling.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple of (batch, seq_len, hidden_size)
        # We take the last 4 layers
        all_hidden_states = outputs.hidden_states

        # Stack the [CLS] token (index 0) from the last 4 layers
        # Shape: (batch_size, 4, hidden_size)
        cls_embeddings = torch.stack(
            [layer_output[:, 0, :] for layer_output in all_hidden_states[-4:]],
            dim=1,
        )

        # Compute normalized weights
        # Shape: (4,)
        weights = torch.softmax(self.layer_weights, dim=0)

        # Weighted sum of the CLS embeddings
        # Reshape weights to (1, 4, 1) for broadcasting
        # Result Shape: (batch_size, hidden_size)
        weighted_output = torch.sum(cls_embeddings * weights.view(1, 4, 1), dim=1)

        # Classification head
        logits = self.fc(weighted_output)

        return logits


class CustomRoberta(nn.Module):
    """
    Branch B: Global Context Transformer
    Backbone: roberta-large
    Head: Mean Pooling (Sequence Average) -> Linear
    """

    def __init__(self, model_name=Config.MODEL_ROBERTA, num_classes=3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.fc = nn.Linear(self.config.hidden_size, num_classes)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass with Mean Pooling.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Last hidden state: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # Expand attention mask to match hidden state dimensions
        # Mask shape: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings across the sequence dimension, ignoring padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Count non-padding tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask

        # Classification head
        logits = self.fc(mean_embeddings)

        return logits
