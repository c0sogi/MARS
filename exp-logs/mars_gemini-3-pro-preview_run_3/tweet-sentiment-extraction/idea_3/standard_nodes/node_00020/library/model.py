import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    TweetModel architecture for Sentiment Extraction.

    Features:
    - Backbone: DeBERTa-v3-base
    - Head: Weighted Layer Pooling (aggregates last N layers)
    - Regularization: Multi-Sample Dropout (ensembles dropout masks)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. Load Configuration and Backbone
        self.hf_config = AutoConfig.from_pretrained(
            config.model_name, output_hidden_states=False
        )

        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=self.hf_config
        )

        # 2. Dropout
        self.dropout = nn.Dropout(config.dropout)

        # 3. Classification Head
        # Projects hidden_size -> 2 (start_logit, end_logit)
        self.fc = nn.Linear(self.hf_config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Shape (batch_size, seq_len)
            attention_mask (torch.Tensor): Shape (batch_size, seq_len)
            token_type_ids (torch.Tensor): Shape (batch_size, seq_len)

        Returns:
            start_logits (torch.Tensor): Shape (batch_size, seq_len)
            end_logits (torch.Tensor): Shape (batch_size, seq_len)
        """
        # 1. Backbone Forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use the final hidden state
        sequence_output = outputs.last_hidden_state

        # 2. Dropout & Classification
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        return start_logits.squeeze(-1), end_logits.squeeze(-1)
