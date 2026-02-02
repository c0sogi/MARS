import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaForMaskedLM, AutoConfig
from library.config import Config


class PointerLocator(nn.Module):
    """
    A custom model that uses a RoBERTa backbone to identify the location of a missing word.
    It functions as a Pointer Network, predicting a score for each token in the sequence.
    The token with the highest score is predicted to be the one immediately PRECEDING the gap.
    """

    def __init__(self, model_name=Config.MODEL_NAME, dropout_rate=0.1):
        super(PointerLocator, self).__init__()

        # Load the pre-trained RoBERTa backbone
        # add_pooling_layer=False because we need the sequence of hidden states, not just the pooled output
        self.roberta = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)

        # The Pointer Head: A linear layer mapping hidden_size -> 1
        # This produces a scalar score for each token position
        self.pointer_head = nn.Linear(self.roberta.config.hidden_size, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids (torch.Tensor): Shape (batch_size, seq_len)
            attention_mask (torch.Tensor): Shape (batch_size, seq_len), 1 for tokens, 0 for padding.
            token_type_ids (torch.Tensor, optional): Shape (batch_size, seq_len). Generally not used for RoBERTa.

        Returns:
            logits (torch.Tensor): Shape (batch_size, seq_len).
                                   Raw scores for each position indicating likelihood of being the pre-gap token.
        """
        # Pass inputs through RoBERTa backbone
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the sequence of hidden states: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # Project to scalar scores: (batch_size, seq_len, 1)
        logits = self.pointer_head(sequence_output)

        # Squeeze the last dimension to get (batch_size, seq_len)
        logits = logits.squeeze(-1)

        # Apply masking to ensure padding tokens are effectively ignored during Softmax/Loss calculation.
        # We set the logits for padded positions (where attention_mask is 0) to a very large negative number.
        # Note: attention_mask is 1 for keep, 0 for remove.
        # (1 - attention_mask) makes padding 1 and tokens 0.
        # We multiply by -1e9 to push padding logits to negative infinity.
        extended_mask = (1.0 - attention_mask) * -1e9
        logits = logits + extended_mask

        return logits


def get_filler_model(model_name=Config.MODEL_NAME):
    """
    Factory function to instantiate the Filler model.
    The Filler is a standard Masked Language Model (MLM) that predicts the token for a <mask`> position.

    Args:
        model_name (str): The name of the pre-trained model to load (default: roberta-base).

    Returns:
        RobertaForMaskedLM: The pre-trained model with an LM head.
    """
    model = RobertaForMaskedLM.from_pretrained(model_name)
    return model
