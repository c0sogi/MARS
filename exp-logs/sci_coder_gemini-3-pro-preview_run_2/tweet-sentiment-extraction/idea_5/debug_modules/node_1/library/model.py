import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TweetModel(nn.Module):
    """
    Neural Network for Tweet Sentiment Extraction.

    Architecture:
    - Backbone: Microsoft DeBERTa-v3-large (Encoder)
    - Head: Linear Layer for Token Classification (Start/End Span Prediction)
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load Transformer Configuration
        self.config = AutoConfig.from_pretrained(
            Config.BERT_PATH, output_hidden_states=True
        )

        # Load Pre-trained Backbone
        self.backbone = AutoModel.from_pretrained(Config.BERT_PATH, config=self.config)

        # Regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # Span Prediction Head
        # Projects hidden states to 2 values per token: [start_logit, end_logit]
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        """
        Initialize weights for the custom linear layer.
        Uses the initializer range specified in the backbone config (typically 0.02).
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of token indices. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Tensor indicating non-padding tokens. Shape: (batch_size, seq_len)

        Returns:
            start_logits (torch.Tensor): Logits for the start index. Shape: (batch_size, seq_len)
            end_logits (torch.Tensor): Logits for the end index. Shape: (batch_size, seq_len)
        """
        # 1. Backbone Encoding
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # 2. Regularization
        sequence_output = self.dropout(sequence_output)

        # 3. Prediction Head
        # logits shape: (batch_size, seq_len, 2)
        logits = self.qa_outputs(sequence_output)

        # 4. Split and Reshape
        # Separate the 2 outputs into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Remove the last dimension to get (batch_size, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
