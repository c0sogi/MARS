import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from library.config import Config


class LocatorModel(nn.Module):
    """
    Stage 1: The Structural Locator.

    This model uses a DeBERTa-v3-Base encoder with a Token Classification head.
    Its purpose is to assign a probability to each token position indicating whether
    a word is missing at that location (specifically, typically marked at the token
    immediately preceding or following the gap).

    Configuration:
        - Backbone: microsoft/deberta-v3-base
        - Head: Token Classification
        - Labels: 2 (0 = No Gap, 1 = Gap)
    """

    def __init__(self, pretrained=True):
        super(LocatorModel, self).__init__()
        model_name = Config.LOCATOR_MODEL_NAME
        num_labels = 2

        if pretrained:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            self.model = AutoModelForTokenClassification.from_config(config)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for the Locator model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            labels (torch.Tensor, optional): Labels for computing the token classification loss.

        Returns:
            TokenClassifierOutput: Object containing loss (if labels provided) and logits.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


class InfillerModel(nn.Module):
    """
    Stage 2: The Semantic In-Filler.

    This model uses a RoBERTa-Large encoder with a Masked Language Modeling (MLM) head.
    Its purpose is to predict the most likely original word for a specific position
    that has been replaced with a <mask> token.

    Configuration:
        - Backbone: roberta-large
        - Head: Masked LM
        - Labels: Vocabulary size (implicit in MLM head)
    """

    def __init__(self, pretrained=True):
        super(InfillerModel, self).__init__()
        model_name = Config.INFILLER_MODEL_NAME

        if pretrained:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_config(config)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for the Infiller model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            labels (torch.Tensor, optional): Labels for computing the masked language modeling loss.

        Returns:
            MaskedLMOutput: Object containing loss (if labels provided) and logits.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


class VerifierModel(nn.Module):
    """
    Stage 3: The Contextual Verifier.

    This model uses a DeBERTa-v3-Large encoder with a Sequence Classification head.
    Its purpose is to act as a discriminator, taking a complete sentence as input
    and outputting a probability score indicating whether the sentence is "Real" (coherent)
    or "Fake" (contains an incorrect insertion).

    Configuration:
        - Backbone: microsoft/deberta-v3-large
        - Head: Sequence Classification
        - Labels: 2 (0 = Fake/Incorrect, 1 = Real/Correct)
    """

    def __init__(self, pretrained=True):
        super(VerifierModel, self).__init__()
        model_name = Config.VERIFIER_MODEL_NAME
        num_labels = 2

        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_config(config)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for the Verifier model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            labels (torch.Tensor, optional): Labels for computing the sequence classification loss.

        Returns:
            SequenceClassifierOutput: Object containing loss (if labels provided) and logits.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
