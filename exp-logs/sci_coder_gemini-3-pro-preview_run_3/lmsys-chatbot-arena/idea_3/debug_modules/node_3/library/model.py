import torch
import torch.nn as nn
from transformers import LongformerModel
from library.config import Config


class LongformerHybridClassifier(nn.Module):
    """
    A hybrid classifier that combines a pre-trained Longformer backbone with
    explicit scalar features (log-transformed lengths) for text classification.
    """

    def __init__(self):
        super(LongformerHybridClassifier, self).__init__()

        # 1. Load Pre-trained Longformer Backbone
        # We use the model name from Config (allenai/longformer-base-4096)
        self.backbone = LongformerModel.from_pretrained(Config.MODEL_NAME)

        # 2. Define Feature Dimensions
        # Longformer base hidden size is usually 768
        self.hidden_size = self.backbone.config.hidden_size

        # We have 3 scalar features: log(len_prompt), log(len_resp_a), log(len_resp_b)
        self.num_scalar_features = 3

        # Combined dimension for the classifier input
        self.combined_dim = self.hidden_size + self.num_scalar_features

        # 3. Define Classification Head
        self.dropout = nn.Dropout(Config.HIDDEN_DROPOUT_PROB)
        self.classifier = nn.Linear(self.combined_dim, Config.NUM_LABELS)

    def forward(
        self, input_ids, attention_mask, global_attention_mask, scalar_features
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor): Local attention mask (batch_size, seq_len).
            global_attention_mask (torch.Tensor): Global attention mask (batch_size, seq_len).
            scalar_features (torch.Tensor): Explicit features (batch_size, 3).

        Returns:
            torch.Tensor: Logits for the 3 classes (batch_size, 3).
        """
        # Pass inputs through the Longformer backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

        # Extract the pooler output
        # This corresponds to the hidden state of the first token (<s>)
        # processed by a linear layer and a tanh activation.
        # Shape: (batch_size, hidden_size)
        pooler_output = outputs.pooler_output

        # Apply dropout
        pooler_output = self.dropout(pooler_output)

        # Concatenate the backbone output with scalar features
        # scalar_features shape: (batch_size, 3)
        # combined_features shape: (batch_size, hidden_size + 3)
        combined_features = torch.cat((pooler_output, scalar_features), dim=1)

        # Pass through the linear classifier to get logits
        logits = self.classifier(combined_features)

        return logits
