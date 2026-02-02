import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MuRILForQA(nn.Module):
    """
    MuRIL-based model for Question Answering.

    This model uses the 'google/muril-base-cased' backbone to generate contextualized
    embeddings for the input sequence (Question + Context). A linear head then projects
    these embeddings to start and end logits for answer span prediction.
    """

    def __init__(self):
        super(MuRILForQA, self).__init__()

        # Load configuration and pre-trained backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.muril = AutoModel.from_pretrained(Config.MODEL_NAME)

        # QA Classification Head
        # Projects hidden states to 2 values: start_logit and end_logit
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the head (optional, but good practice)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        """Initialize the weights of the linear head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor): Segment token indices to indicate first and second portions of the inputs.
                                           Crucial for MuRIL to distinguish Question (0) from Context (1).

        Returns:
            start_logits (torch.Tensor): Logits for the start position of the answer. Shape: (batch_size, seq_len)
            end_logits (torch.Tensor): Logits for the end position of the answer. Shape: (batch_size, seq_len)
        """

        # Pass inputs to the MuRIL backbone
        # We explicitly pass token_type_ids as MuRIL (BERT-based) relies on them for segment embedding
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # sequence_output shape: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]

        # Project to QA logits
        # logits shape: (batch_size, sequence_length, 2)
        logits = self.qa_outputs(sequence_output)

        # Split logits into start and end
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch_size, sequence_length)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
