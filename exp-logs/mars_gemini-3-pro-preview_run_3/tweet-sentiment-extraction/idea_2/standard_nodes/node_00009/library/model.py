import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TweetModel(nn.Module):
    """
    Neural Network for Tweet Sentiment Extraction.

    Architecture:
    1. Backbone: DeBERTa-v3-base (pretrained).
    2. Head: Multi-Sample Dropout + Linear Layer.

    The model predicts the start and end logits for the selected text span.
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_NAME, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Multi-Sample Dropout settings
        self.use_msd = Config.USE_MULTI_SAMPLE_DROPOUT
        self.dropout_rates = Config.DROPOUT_RATES if self.use_msd else [0.1]

        # Dropout Layers
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_rates])

        # Classification Head
        # Output size is 2: one for start logit, one for end logit
        self.classifier = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the new head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head following standard BERT initialization.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.

        Returns:
            start_logits (torch.Tensor): Logits for the start index (Batch, Seq_Len).
            end_logits (torch.Tensor): Logits for the end index (Batch, Seq_Len).
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden state: (Batch, Seq_Len, Hidden_Size)
        sequence_output = outputs.last_hidden_state

        # Apply Multi-Sample Dropout
        # We calculate logits for each dropout mask and then average them
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout
            out = dropout(sequence_output)
            # Project to 2 dimensions (start/end)
            logits = self.classifier(out)
            logits_list.append(logits)

        # Stack logits: (Num_Drops, Batch, Seq_Len, 2)
        stacked_logits = torch.stack(logits_list, dim=0)

        # Average over the dropout dimension: (Batch, Seq_Len, 2)
        mean_logits = torch.mean(stacked_logits, dim=0)

        # Split into start and end logits
        # split returns a tuple of tensors. Each has shape (Batch, Seq_Len, 1)
        start_logits, end_logits = mean_logits.split(1, dim=-1)

        # Squeeze the last dimension to get (Batch, Seq_Len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
