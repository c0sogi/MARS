import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WideLinear(nn.Module):
    """
    A simple one-layer neural network (Neural Logistic Regression) for the Wide component.
    Maps high-dimensional sparse TF-IDF features directly to output logits.
    """

    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): Dimension of the input features (vocabulary size).
            output_dim (int): Dimension of the output (number of tags).
        """
        super(WideLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, output_dim).
        """
        return self.linear(x)


class DeepTransformer(nn.Module):
    """
    A Transformer-based model for the Deep component.
    Wraps a pre-trained DistilRoBERTa model with a classification head.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_TOP_TAGS):
        """
        Args:
            model_name (str): Name of the pre-trained model to load (e.g., 'distilroberta-base').
            num_classes (int): Number of output classes (tags).
        """
        super(DeepTransformer, self).__init__()

        # Load configuration and pre-trained backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        # Classification head
        # We use a standard dropout followed by a linear projection
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention masks of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Pass inputs through the transformer backbone
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the representation of the [CLS] token (first token in sequence)
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        cls_token_state = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and projection
        x = self.dropout(cls_token_state)
        logits = self.classifier(x)

        return logits
