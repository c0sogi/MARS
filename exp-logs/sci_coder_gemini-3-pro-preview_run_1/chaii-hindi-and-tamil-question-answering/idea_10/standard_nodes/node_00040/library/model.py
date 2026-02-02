import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MultiTaskXLMR(nn.Module):
    """
    Multi-Task XLM-Roberta Large model for Question Answering and Relevance Classification.

    Architecture:
    1. Backbone: xlm-roberta-large
    2. Span Head: Linear layer to predict start and end logits for answer extraction.
    3. Relevance Head: Linear layer on the [CLS] token to predict if the window contains the answer.
    """

    def __init__(self, pretrained_model_name=Config.MODEL_CHECKPOINT):
        super(MultiTaskXLMR, self).__init__()

        # Load configuration and backbone
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # 1. Span Head: Predicts start (0) and end (1) logits for each token
        # Input: (Batch, Seq_Len, Hidden) -> Output: (Batch, Seq_Len, 2)
        self.span_head = nn.Linear(self.config.hidden_size, 2)

        # 2. Relevance Head: Predicts if the context window contains the answer
        # Input: (Batch, Hidden) [CLS token] -> Output: (Batch, 1)
        self.relevance_head = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the new heads
        self._init_weights(self.span_head)
        self._init_weights(self.relevance_head)

    def _init_weights(self, module):
        """
        Initialize the weights of the custom heads using the same standard deviation
        as the pre-trained model's initializer.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices.

        Returns:
            start_logits (torch.Tensor): Logits for the start position (Batch, Seq_Len)
            end_logits (torch.Tensor): Logits for the end position (Batch, Seq_Len)
            relevance_logits (torch.Tensor): Logits for window relevance (Batch, 1)
        """
        # Pass inputs through the backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Sequence output: (Batch, Seq_Len, Hidden_Size)
        sequence_output = outputs.last_hidden_state

        # CLS token embedding (Index 0 for XLM-R): (Batch, Hidden_Size)
        cls_embedding = sequence_output[:, 0, :]

        # --- Span Prediction ---
        # (Batch, Seq_Len, Hidden) -> (Batch, Seq_Len, 2)
        span_logits = self.span_head(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = span_logits.split(1, dim=-1)

        # Squeeze the last dimension to get (Batch, Seq_Len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # --- Relevance Prediction ---
        # (Batch, Hidden) -> (Batch, 1)
        relevance_logits = self.relevance_head(cls_embedding)

        return start_logits, end_logits, relevance_logits
