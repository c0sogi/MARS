import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class DebertaWithMSD(nn.Module):
    """
    DeBERTa-v3 architecture with Weighted Layer Pooling and Multi-Sample Dropout.

    Architecture:
    1. Backbone: microsoft/deberta-v3-large
    2. Pooling: Weighted average of the [CLS] tokens from the last 4 hidden layers.
       Weights are learnable parameters.
    3. Regularization: Multi-Sample Dropout (MSD). The pooled embedding is passed
       through multiple dropout layers with different masks/rates.
    4. Head: A single Linear layer shared across all dropout samples.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=3,
        dropout_rates=Config.DROPOUT_RATES,
    ):
        super().__init__()

        # 1. Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        # We need hidden states for Weighted Layer Pooling
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # 2. Weighted Layer Pooling Setup
        # We aggregate the last 4 layers
        self.n_layers_to_pool = 4
        # Initialize weights to be equal (will be softmaxed in forward pass)
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers_to_pool))

        # 3. Multi-Sample Dropout Setup
        # Create a list of dropout modules based on the rates in Config
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

        # 4. Classification Head
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize the new head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, target=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token indices (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).
            target (torch.Tensor, optional): Labels (not used in forward, handled by loss fn).

        Returns:
            torch.Tensor: Logits (batch_size, num_classes).
        """
        # 1. Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Weighted Layer Pooling
        # outputs.hidden_states is a tuple of (batch, seq_len, hidden)
        # We take the last n layers
        all_hidden_states = outputs.hidden_states
        last_four_layers = all_hidden_states[-self.n_layers_to_pool :]

        # Extract [CLS] token (index 0) from each of these layers
        # Stack to shape: (batch_size, n_layers, hidden_size)
        cls_embeddings = torch.stack(
            [layer[:, 0, :] for layer in last_four_layers], dim=1
        )

        # Compute normalized weights
        weights = F.softmax(self.layer_weights, dim=0)

        # Reshape weights for broadcasting: (1, n_layers, 1)
        weights_expanded = weights.view(1, self.n_layers_to_pool, 1)

        # Compute weighted average: Sum(weight * embedding)
        # Result shape: (batch_size, hidden_size)
        weighted_cls_embedding = torch.sum(cls_embeddings * weights_expanded, dim=1)

        # 3. Multi-Sample Dropout & Classification
        # Pass the embedding through each dropout mask, classify, and collect logits
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout
            dropped_embedding = dropout(weighted_cls_embedding)
            # Classify
            logits = self.fc(dropped_embedding)
            logits_list.append(logits)

        # 4. Average the logits
        # Stack to (n_samples, batch, num_classes) then mean over dim 0
        mean_logits = torch.mean(torch.stack(logits_list), dim=0)

        return mean_logits
