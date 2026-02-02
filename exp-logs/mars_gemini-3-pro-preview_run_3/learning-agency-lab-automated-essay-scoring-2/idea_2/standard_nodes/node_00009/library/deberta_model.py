import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class EssayRegressor(nn.Module):
    """
    A PyTorch module for essay scoring based on the DeBERTa-v3 architecture.

    This model serves two purposes:
    1. Regression: Predicts a scalar score for an essay (Stage 1).
    2. Feature Extraction: Generates dense vector embeddings for essay chunks (Stage 2).
    """

    def __init__(self, model_path=None, config_path=None):
        """
        Initialize the EssayRegressor.

        Args:
            model_path (str, optional): Path or HuggingFace ID for the model.
                                        Defaults to Config.MODEL_NAME.
            config_path (str, optional): Path to specific config if different.
        """
        super(EssayRegressor, self).__init__()

        target_model = model_path if model_path else Config.MODEL_NAME

        # Load Configuration
        self.config = AutoConfig.from_pretrained(target_model)

        # Load Pre-trained Backbone
        # We use AutoModel to get the raw hidden states
        self.backbone = AutoModel.from_pretrained(target_model, config=self.config)

        # Regression Head
        # Maps the hidden size (e.g., 768 for base) to a single scalar score
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize the head weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the regression head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, return_embedding=False):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            return_embedding (bool): If True, returns the [CLS] embedding instead of the score.

        Returns:
            torch.Tensor:
                - If return_embedding is False: Scalar scores of shape (batch_size, 1).
                - If return_embedding is True: Embeddings of shape (batch_size, hidden_size).
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Extract [CLS] token representation (index 0)
        # Shape: (batch_size, hidden_size)
        cls_embeddings = last_hidden_state[:, 0, :]

        if return_embedding:
            return cls_embeddings

        # Pass through regression head
        # Shape: (batch_size, 1)
        logits = self.fc(cls_embeddings)

        return logits
