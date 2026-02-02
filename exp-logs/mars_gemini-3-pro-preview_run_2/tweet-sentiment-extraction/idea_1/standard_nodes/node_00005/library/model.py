import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import MODEL_NAME, DROPOUT


class TransformerPointerNetwork(nn.Module):
    """
    A Transformer-based Pointer Network for sentiment span extraction.
    Uses a pre-trained model (e.g., RoBERTa) as the encoder.
    """

    def __init__(self):
        super(TransformerPointerNetwork, self).__init__()

        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.roberta = AutoModel.from_pretrained(MODEL_NAME, config=self.config)

        self.dropout = nn.Dropout(DROPOUT)

        # Two outputs: start logic and end logit
        self.start_head = nn.Linear(self.config.hidden_size, 1)
        self.end_head = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for heads
        nn.init.xavier_uniform_(self.start_head.weight)
        nn.init.xavier_uniform_(self.end_head.weight)
        nn.init.constant_(self.start_head.bias, 0)
        nn.init.constant_(self.end_head.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
        """
        # Transformer Output
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Last hidden state: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Predict Start and End logits
        start_logits = self.start_head(sequence_output).squeeze(-1)
        end_logits = self.end_head(sequence_output).squeeze(-1)

        return start_logits, end_logits
