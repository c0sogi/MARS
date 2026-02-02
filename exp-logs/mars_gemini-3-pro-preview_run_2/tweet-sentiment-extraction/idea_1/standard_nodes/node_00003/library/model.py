import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import MODEL_NAME, DROPOUT


class TweetModel(nn.Module):
    """
    Transformer-based model for sentiment span extraction.
    Uses RoBERTa as the encoder and two linear heads for start/end prediction.
    Cite solution_lesson_node_00001: Replaces Bi-GRU with Transformer to improve semantic understanding.
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load pre-trained transformer
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)

        # Hidden size of the transformer (768 for base)
        self.hidden_size = self.roberta.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(DROPOUT)

        # Output heads (QA style)
        self.start_head = nn.Linear(self.hidden_size, 1)
        self.end_head = nn.Linear(self.hidden_size, 1)

        # Weight initialization for heads
        nn.init.xavier_uniform_(self.start_head.weight)
        nn.init.xavier_uniform_(self.end_head.weight)

    def forward(self, input_ids, sentiment_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            sentiment_ids: Unused in forward (encoded in input_ids via tokenizer)
            attention_mask: (batch_size, seq_len)

        Returns:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
        """
        # Pass through Transformer
        # outputs.last_hidden_state: (batch_size, seq_len, hidden_size)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply dropout
        out = self.dropout(last_hidden_state)

        # Predict logits
        start_logits = self.start_head(out).squeeze(-1)
        end_logits = self.end_head(out).squeeze(-1)

        return start_logits, end_logits
