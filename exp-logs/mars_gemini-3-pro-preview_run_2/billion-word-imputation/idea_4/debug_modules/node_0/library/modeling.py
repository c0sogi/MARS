import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig
from library.config import Config


class GapLocatorModel(nn.Module):
    """
    Stage 1 Model: Gap Locator.

    Wraps a pre-trained Transformer encoder with a token classification head.
    Predicts the probability of a missing word existing immediately after each token.
    """

    def __init__(
        self, model_name: str = Config.LOCATOR_MODEL, dropout_rate: float = 0.1
    ):
        """
        Args:
            model_name (str): Name of the pre-trained model to load.
            dropout_rate (float): Dropout probability for the classification head.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)

        # Project hidden size to 1 logit (Binary Classification: Gap vs No Gap)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        # Loss function: BCE with Logits
        # reduction='none' allows us to mask out padding tokens manually
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids (torch.Tensor): (Batch, Seq_Len)
            attention_mask (torch.Tensor): (Batch, Seq_Len)
            labels (torch.Tensor, optional): (Batch, Seq_Len) - 0.0 or 1.0

        Returns:
            dict: {"logits": torch.Tensor, "loss": torch.Tensor (scalar) or None}
        """
        # 1. Base Encoder Pass
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (Batch, Seq_Len, Hidden)

        # 2. Classification Head
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (Batch, Seq_Len, 1)
        logits = logits.squeeze(-1)  # (Batch, Seq_Len)

        loss = None
        if labels is not None:
            # 3. Loss Calculation
            # Compute element-wise loss
            per_token_loss = self.loss_fct(logits, labels)

            # Mask out padding tokens using attention_mask
            # attention_mask is 1 for tokens, 0 for padding
            active_loss_mask = attention_mask.view(-1) == 1
            active_loss = per_token_loss.view(-1)[active_loss_mask]

            # Average loss only over non-padding tokens
            loss = active_loss.mean()

        return {"logits": logits, "loss": loss}


class InFillerModel(nn.Module):
    """
    Stage 2 Model: In-Filler.

    Wraps a pre-trained AutoModelForMaskedLM.
    Predicts the token ID for masked positions.
    """

    def __init__(self, model_name: str = Config.INFILLER_MODEL):
        """
        Args:
            model_name (str): Name of the pre-trained model to load.
        """
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids (torch.Tensor): (Batch, Seq_Len)
            attention_mask (torch.Tensor): (Batch, Seq_Len)
            labels (torch.Tensor, optional): (Batch, Seq_Len) - Token IDs

        Returns:
            MaskedLMOutput (acts like dict): keys 'loss', 'logits'
        """
        # AutoModelForMaskedLM handles the MLM head and CrossEntropyLoss internally
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
