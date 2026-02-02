import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class CustomDeberta(nn.Module):
    """
    Custom DeBERTa model wrapper for Phrase Similarity Regression.

    This class wraps the HuggingFace AutoModelForSequenceClassification to allow
    for custom handling of Atomic Context Embeddings. Specifically, it handles
    the resizing of the token embedding layer to accommodate new special tokens
    representing CPC contexts.
    """

    def __init__(self, config, tokenizer):
        """
        Initializes the model and resizes embeddings.

        Args:
            config: Configuration object containing model_name and num_classes.
            tokenizer: The tokenizer instance, which must already have the
                       unique context tokens added to its vocabulary.
        """
        super().__init__()

        # Load the pre-trained backbone with a regression head.
        # num_labels=1 triggers the model to use MSELoss when labels are provided.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=config.num_classes
        )

        # Resize the model's embedding layer to match the tokenizer's new size.
        # This allocates new, randomly initialized vectors for the added
        # Atomic Context tokens, which will be learned during training.
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
            labels (torch.Tensor, optional): Labels for computing the Mean Squared Error loss.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput:
                Contains 'loss' (if labels are provided) and 'logits' (the regression score).
        """
        # Pass inputs to the underlying HuggingFace model.
        # The model handles the regression logic and loss computation internally.
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        return output
