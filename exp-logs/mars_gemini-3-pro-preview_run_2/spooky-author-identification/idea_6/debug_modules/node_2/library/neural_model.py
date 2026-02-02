import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class DebertaMultiView(nn.Module):
    """
    Custom DeBERTa architecture with Multi-View Head.

    This model extracts hidden states from the last 4 layers of the backbone,
    computes a learnable weighted average, and applies both Mean and Max pooling
    to capture diverse linguistic features.

    Architecture:
    1. Backbone: microsoft/deberta-v3-large
    2. Fusion: Weighted sum of last 4 layers (weights are learnable parameters).
    3. Pooling: Concatenation of Mean Pooling (global style) and Max Pooling (salient features).
    4. Head: Linear classification layer.
    """

    def __init__(self):
        super(DebertaMultiView, self).__init__()

        # 1. Load Configuration and Backbone
        # We enable output_hidden_states to access intermediate layers for fusion
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_NAME, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # 2. Learnable Weights for Layer Fusion
        # We use 4 scalar weights corresponding to the last 4 layers.
        # Initialized to 0.0 so softmax yields equal weighting (0.25) at start.
        self.layer_weights = nn.Parameter(torch.zeros(4))

        # 3. Classification Head
        # Input dimension is hidden_size * 2 due to concatenation of Mean and Max pooling
        self.fc = nn.Linear(self.config.hidden_size * 2, Config.NUM_LABELS)

        # Initialize the custom head weights
        self._init_custom_weights()

    def _init_custom_weights(self):
        """
        Applies Xavier initialization to the linear classification head.
        """
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            labels (torch.Tensor, optional): Labels for computing the CrossEntropy loss.

        Returns:
            dict: A dictionary containing 'logits' and optionally 'loss'.
        """
        # 1. Backbone Forward Pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple of (embeddings, layer_1, ..., layer_N)
        # We extract the last 4 layers for our multi-view fusion.
        all_states = outputs.hidden_states

        # Stack the last 4 layers -> Shape: (Batch, 4, Seq_Len, Hidden)
        stacked_layers = torch.stack(all_states[-4:], dim=1)

        # 2. Weighted Layer Fusion
        # Compute normalized weights using softmax -> Shape: (4,)
        weights = torch.softmax(self.layer_weights, dim=0)

        # Reshape for broadcasting -> Shape: (1, 4, 1, 1)
        weights = weights.view(1, 4, 1, 1)

        # Compute weighted sum across the layer dimension -> Shape: (Batch, Seq_Len, Hidden)
        weighted_output = (stacked_layers * weights).sum(dim=1)

        # 3. Dual Pooling (Mean + Max)
        # Expand attention mask to match hidden size -> Shape: (Batch, Seq_Len, Hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(weighted_output.size()).float()
        )

        # Mean Pooling: Sum of valid tokens / Count of valid tokens
        sum_embeddings = torch.sum(weighted_output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Prevent division by zero
        mean_pooled = sum_embeddings / sum_mask

        # Max Pooling: Max of valid tokens
        # We replace padding tokens with a very small number (-1e9) so they are not selected by max()
        weighted_output_masked = weighted_output.clone()
        weighted_output_masked[input_mask_expanded == 0] = -1e9
        max_pooled, _ = torch.max(weighted_output_masked, dim=1)

        # Concatenate the two representations -> Shape: (Batch, Hidden * 2)
        fused_vector = torch.cat([mean_pooled, max_pooled], dim=1)

        # 4. Classification
        logits = self.fc(fused_vector)

        # 5. Loss Calculation (if labels provided)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
