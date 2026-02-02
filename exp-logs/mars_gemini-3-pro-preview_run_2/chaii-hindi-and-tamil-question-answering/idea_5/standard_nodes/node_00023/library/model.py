import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MuRILForQA(nn.Module):
    """
    MuRIL-based model for Question Answering.
    """

    def __init__(self):
        super(MuRILForQA, self).__init__()

        # Load configuration and pre-trained backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.muril = AutoModel.from_pretrained(Config.MODEL_NAME)

        # QA Classification Head
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the head
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
        MuRIL (BERT-based) uses token_type_ids.
        """

        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # sequence_output shape: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]

        # Project to QA logits
        logits = self.qa_outputs(sequence_output)

        # Split logits into start and end
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch_size, sequence_length)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
