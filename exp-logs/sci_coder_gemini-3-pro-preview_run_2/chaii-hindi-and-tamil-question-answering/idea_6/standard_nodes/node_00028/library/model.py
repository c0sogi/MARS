import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class XLMROBERTAForQA(nn.Module):
    """
    XLM-RoBERTa model with a linear layer for Question Answering.

    This model wraps the 'xlm-roberta-base' backbone. It projects the
    last hidden states to start and end logits for span prediction.
    It intentionally ignores token_type_ids as XLM-R does not use them.
    """

    def __init__(self, pretrained=True):
        super(XLMROBERTAForQA, self).__init__()

        # Load configuration from the checkpoint defined in Config
        self.config = AutoConfig.from_pretrained(Config.MODEL_CHECKPOINT)

        # Load the backbone
        if pretrained:
            self.roberta = AutoModel.from_pretrained(
                Config.MODEL_CHECKPOINT, config=self.config
            )
        else:
            self.roberta = AutoModel.from_config(self.config)

        # QA Head: Projects hidden size to 2 (start_logits, end_logits)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the head
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        """
        Initialize weights using the standard BERT/RoBERTa initialization scheme.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor): Segment token indices. Ignored for XLM-RoBERTa.

        Returns:
            tuple: (start_logits, end_logits)
        """
        # XLM-RoBERTa does not use token_type_ids (segment embeddings).
        # We pass only input_ids and attention_mask.
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # sequence_output shape: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Project to logits
        # logits shape: (batch_size, sequence_length, 2)
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        # start_logits shape: (batch_size, sequence_length)
        # end_logits shape: (batch_size, sequence_length)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
