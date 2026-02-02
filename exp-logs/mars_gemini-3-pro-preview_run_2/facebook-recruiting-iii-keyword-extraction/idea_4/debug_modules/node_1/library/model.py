import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class DistilRobertaForTagging(nn.Module):
    """
    DistilRoBERTa-based model for multi-label classification.

    Architecture:
    1. DistilRoBERTa Backbone (pre-trained)
    2. Dropout Layer
    3. Linear Classification Head
    """

    def __init__(self, config=Config):
        """
        Initializes the model architecture.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super(DistilRobertaForTagging, self).__init__()

        # Load configuration from pre-trained model name
        self.model_config = AutoConfig.from_pretrained(config.model_name)

        # Initialize the pre-trained backbone
        self.roberta = AutoModel.from_pretrained(
            config.model_name, config=self.model_config
        )

        # Classification Head
        # We use the hidden size from the config (usually 768 for distilroberta-base)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.model_config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                                      Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
                                           Shape: (batch_size, sequence_length)

        Returns:
            logits (torch.Tensor): Raw output scores for each label.
                                   Shape: (batch_size, num_labels)
        """
        # Pass inputs through the backbone
        # outputs is a BaseModelOutputWithPoolingAndCrossAttentions object
        # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token representation.
        # In RoBERTa/DistilRoBERTa, the [CLS] token is at index 0.
        cls_token_state = outputs.last_hidden_state[:, 0, :]

        # Apply Dropout
        x = self.dropout(cls_token_state)

        # Apply Linear Layer to get logits
        logits = self.classifier(x)

        return logits
