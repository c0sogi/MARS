import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    Tweet Sentiment Extraction Model based on DeBERTa-v3-Large.

    Features:
    - Backbone: microsoft/deberta-v3-large
    - Aggregation: Weighted Layer Pooling (Last 4 layers)
    - Head: Linear projection to start/end logits
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load configuration and ensure hidden states are outputted
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_NAME, output_hidden_states=True
        )

        # Load the pre-trained backbone
        self.model = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Weighted Layer Pooling weights
        # We initialize them to be roughly equal, allowing the model to learn the best mix
        self.layer_weights = nn.Parameter(torch.tensor([1 / 4] * 4, dtype=torch.float))

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Final prediction head: projects hidden_size -> 2 (start_logit, end_logit)
        self.fc = nn.Linear(self.config.hidden_size, 2)

        # Initialize the head weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the custom head layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices.

        Returns:
            start_logits (torch.Tensor): Logits for the start index.
            end_logits (torch.Tensor): Logits for the end index.
        """

        # Pass through the backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # outputs.hidden_states is a tuple of tensors (one for the output of the embeddings + one for the output of each layer)
        # We want the last 4 layers.
        all_hidden_states = outputs.hidden_states

        # Stack the last 4 hidden states: (Batch, SeqLen, HiddenSize, 4)
        stacked_layers = torch.stack(all_hidden_states[-4:], dim=-1)

        # Apply Softmax to learnable weights to ensure they sum to 1
        weights = F.softmax(self.layer_weights, dim=0)

        # Reshape weights for broadcasting: (1, 1, 1, 4)
        weights = weights.view(1, 1, 1, 4)

        # Compute weighted sum: (Batch, SeqLen, HiddenSize)
        weighted_output = (stacked_layers * weights).sum(dim=-1)

        # Apply dropout
        sequence_output = self.dropout(weighted_output)

        # Project to logits: (Batch, SeqLen, 2)
        logits = self.fc(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension: (Batch, SeqLen)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
