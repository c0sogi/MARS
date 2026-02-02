import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM
from library.config import Config


class LocatorModel(nn.Module):
    """
    Stage 1: The Syntactic Locator.

    Uses a DeBERTa-v3 backbone with a custom token classification head.
    Predicts the likelihood of a missing word occurring *after* each token in the sequence.
    """

    def __init__(self):
        super(LocatorModel, self).__init__()
        # Load Configuration and Backbone
        # DeBERTa-v3 is chosen for its disentangled attention, superior for relative positioning
        self.config = AutoConfig.from_pretrained(Config.LOCATOR_MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.LOCATOR_MODEL_NAME)

        # Token Classification Head
        # We project the hidden size (768 for base) to 1 logit per token.
        # This logit represents the score for class "1" (Gap Here).
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights of the head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights of the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass for the Locator.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len), indicating gap probability.
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract last hidden state: (Batch, Seq_Len, Hidden_Size)
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # Project to logits: (Batch, Seq_Len, 1)
        logits = self.classifier(sequence_output)

        # Squeeze to (Batch, Seq_Len) for compatibility with BCEWithLogitsLoss
        return logits.squeeze(-1)


class InfillerModel(nn.Module):
    """
    Stage 2: The Semantic Validator (In-Filler).

    Uses a RoBERTa-Large backbone with a Masked Language Modeling (MLM) head.
    Predicts the token that should fill a <mask> token.
    """

    def __init__(self):
        super(InfillerModel, self).__init__()
        # Load pre-trained model with LM head
        # RoBERTa-Large provides high semantic capacity
        self.model = AutoModelForMaskedLM.from_pretrained(Config.INFILLER_MODEL_NAME)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for the Infiller.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding.
            labels (torch.Tensor, optional): Labels for computing the masked language modeling loss.

        Returns:
            transformers.modeling_outputs.MaskedLMOutput: Contains loss (if labels provided) and logits.
        """
        # AutoModelForMaskedLM handles the forward pass, including loss calculation if labels are provided
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output
