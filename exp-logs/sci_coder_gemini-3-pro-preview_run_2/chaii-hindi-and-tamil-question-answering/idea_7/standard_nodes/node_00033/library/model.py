import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class XLMRobertaForMultiTaskQA(nn.Module):
    """
    Multi-Task XLM-RoBERTa model for Question Answering.

    This architecture includes:
    1. Backbone: XLM-RoBERTa Large
    2. QA Head: Predicts start and end token positions.
    3. Answerability Head: Predicts whether the span contains an answer (binary classification).
    """

    def __init__(self, model_name=Config.MODEL_NAME):
        super(XLMRobertaForMultiTaskQA, self).__init__()

        # Load configuration and backbone model
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)

        # QA Head: Linear layer to predict start and end logits (Hidden -> 2)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Answerability Head: Linear layer on CLS token (Hidden -> 1)
        self.answerability_classifier = nn.Linear(self.config.hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Initialize weights for the custom heads
        self._init_weights(self.qa_outputs)
        self._init_weights(self.answerability_classifier)

    def _init_weights(self, module):
        """Initialize the weights of the linear layers using config settings."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs. Shape: (Batch, Seq_Len)
            attention_mask (torch.Tensor): Attention mask. Shape: (Batch, Seq_Len)

        Returns:
            start_logits (torch.Tensor): Logits for start position. Shape: (Batch, Seq_Len)
            end_logits (torch.Tensor): Logits for end position. Shape: (Batch, Seq_Len)
            answerability_logit (torch.Tensor): Logit for answerability. Shape: (Batch,)
        """
        # Pass inputs through the backbone
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, return_dict=True
        )

        sequence_output = outputs.last_hidden_state  # Shape: (Batch, Seq_Len, Hidden)

        # Extract the CLS token representation (index 0 for XLM-R)
        cls_output = sequence_output[:, 0, :]  # Shape: (Batch, Hidden)

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        cls_output = self.dropout(cls_output)

        # 1. Compute QA Logits
        logits = self.qa_outputs(sequence_output)  # Shape: (Batch, Seq_Len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # Shape: (Batch, Seq_Len)
        end_logits = end_logits.squeeze(-1)  # Shape: (Batch, Seq_Len)

        # 2. Compute Answerability Logit
        answerability_logit = self.answerability_classifier(
            cls_output
        )  # Shape: (Batch, 1)
        answerability_logit = answerability_logit.squeeze(-1)  # Shape: (Batch,)

        return start_logits, end_logits, answerability_logit
