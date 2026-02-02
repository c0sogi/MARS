import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).
    Applies multiple dropout masks with different rates to the input,
    passes them through a shared linear layer, and averages the outputs.
    This acts as an internal ensemble, smoothing the loss landscape and improving generalization.
    """

    def __init__(self, in_features, out_features, dropout_rates):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(rate) for rate in dropout_rates])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: [batch_size, hidden_dim]
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.linear(dropout(x))
            else:
                output += self.linear(dropout(x))

        # Average the outputs
        return output / len(self.dropouts)


class DebertaV3Regressor(nn.Module):
    """
    DeBERTa-v3-Large based regression model for Semantic Similarity.
    Uses a Cross-Encoder architecture with a Multi-Sample Dropout head.
    """

    def __init__(self, model_name=Config.model_name, pretrained=True):
        super().__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Enable gradient checkpointing for memory efficiency
        self.config.gradient_checkpointing = True

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Explicitly enable gradient checkpointing on the backbone instance
        # This is crucial for fitting DeBERTa-Large in memory with reasonable batch sizes
        self.backbone.gradient_checkpointing_enable()

        # The hidden size of the transformer output (1024 for Large)
        self.hidden_size = self.config.hidden_size

        # Regression Head
        if Config.use_msd:
            self.head = MultiSampleDropout(
                in_features=self.hidden_size,
                out_features=1,
                dropout_rates=Config.msd_rates,
            )
        else:
            # Fallback to standard head if MSD is disabled in config
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.hidden_size, 1))

        # Initialize weights of the head
        self._init_weights(self.head)

    def _init_weights(self, module):
        """
        Initialize the weights of the custom head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Sequential, MultiSampleDropout)):
            for sub_module in module.children():
                self._init_weights(sub_module)
        elif isinstance(module, nn.ModuleList):
            for sub_module in module:
                self._init_weights(sub_module)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.

        Returns:
            torch.Tensor: Predicted similarity scores of shape [batch_size].
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # Extract CLS token representation (first token)
        # Shape: [batch_size, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 0, :]

        # Pass through regression head
        # Shape: [batch_size, 1]
        logits = self.head(cls_embeddings)

        # Squeeze to shape [batch_size] for compatibility with loss functions and metric calculation
        return logits.squeeze(-1)
