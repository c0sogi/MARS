import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TransformerClassifier(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, dropout_rate=0.1):
        """
        Initializes the Transformer-based classifier.

        Args:
            model_name (str): The name of the pre-trained model to load.
            dropout_rate (float): The dropout probability for the classification head.
        """
        super(TransformerClassifier, self).__init__()

        # Load the configuration and the pre-trained backbone model
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Define the classification head
        # We use the hidden size from the config (e.g., 768 for distilroberta-base)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Tensor of attention masks. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Logits for the binary classification task. Shape: (batch_size, 1)
        """
        # Pass inputs through the transformer backbone
        # The output object contains 'last_hidden_state' among other fields
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the representation of the [CLS] token.
        # For DistilRoBERTa (and BERT-like models), this corresponds to the first token
        # of the last hidden state sequence.
        # Shape: (batch_size, hidden_size)
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Apply dropout for regularization
        x = self.dropout(cls_token)

        # Pass through the linear layer to get unnormalized logits
        logits = self.classifier(x)

        return logits
