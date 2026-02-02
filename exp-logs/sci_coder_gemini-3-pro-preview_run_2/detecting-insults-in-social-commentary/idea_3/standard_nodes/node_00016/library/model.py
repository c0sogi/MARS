import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class InsultModel(nn.Module):
    """
    Insult Detection Model based on DeBERTa-v3 with Mean Pooling and Multi-Sample Dropout.
    """

    def __init__(self):
        super().__init__()
        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Multi-Sample Dropout
        # We create multiple dropout layers to be applied in parallel (conceptually)
        # The outputs are then averaged. This acts as an ensemble within the model.
        # Using a standard dropout rate of 0.2 for the head.
        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(0.2) for _ in range(Config.dropout_samples)]
        )

        # Shared Classification Head
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (Batch, Seq_Len)
            attention_mask (torch.Tensor): Attention mask (Batch, Seq_Len)

        Returns:
            torch.Tensor: Logits (Batch,)
        """
        # 1. Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (Batch, Seq_Len, Hidden)

        # 2. Mean Pooling
        # Expand mask to match hidden size: (Batch, Seq_Len) -> (Batch, Seq_Len, Hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask to get count of valid tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask  # (Batch, Hidden)

        # 3. Multi-Sample Dropout
        # Apply multiple dropout masks and average the predictions
        logits_list = []
        for dropout_op in self.dropout_ops:
            # Apply dropout
            dropped_embeddings = dropout_op(mean_embeddings)
            # Apply linear layer
            logits = self.fc(dropped_embeddings)
            logits_list.append(logits)

        # Stack logits: (Samples, Batch, 1)
        stacked_logits = torch.stack(logits_list, dim=0)

        # Average across samples: (Batch, 1)
        mean_logits = torch.mean(stacked_logits, dim=0)

        # Squeeze to match target shape (Batch,)
        return mean_logits.squeeze(-1)
