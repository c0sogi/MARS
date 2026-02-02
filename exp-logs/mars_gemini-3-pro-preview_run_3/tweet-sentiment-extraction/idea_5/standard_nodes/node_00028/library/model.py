import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    DeBERTa-v3-Large with Dual-Head Multi-Task Learning.

    Architecture:
    1. Backbone: microsoft/deberta-v3-large
    2. Pointer Head: Predicts start and end indices (Boundary detection).
    3. Content Head: Predicts token-level binary mask (Semantic segmentation).
    """

    def __init__(self, model_path=None):
        """
        Initializes the model architecture.

        Args:
            model_path (str, optional): Path or HuggingFace ID of the model.
                                        Defaults to Config.MODEL_PATH.
        """
        super(TweetModel, self).__init__()

        path = model_path if model_path else Config.MODEL_PATH

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(path, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(path, config=self.config)

        # Regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # Pointer Head: [Hidden Size] -> 2 (Start Logit, End Logit)
        self.pointer_head = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for custom heads
        self._init_weights(self.pointer_head)

    def _init_weights(self, module):
        """
        Initialize weights for the new linear layers using standard transformer initialization.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs (Batch, Seq_Len).
            attention_mask (torch.Tensor): Tensor of attention masks (Batch, Seq_Len).

        Returns:
            start_logits (torch.Tensor): Logits for start position (Batch, Seq_Len).
            end_logits (torch.Tensor): Logits for end position (Batch, Seq_Len).
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = (
            outputs.last_hidden_state
        )  # Shape: (Batch, Seq_Len, Hidden_Size)

        # Apply dropout
        out = self.dropout(last_hidden_state)

        # Pointer Head Predictions
        pointer_logits = self.pointer_head(out)  # Shape: (Batch, Seq_Len, 2)
        start_logits, end_logits = pointer_logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)  # Shape: (Batch, Seq_Len)
        end_logits = end_logits.squeeze(-1)  # Shape: (Batch, Seq_Len)

        return start_logits, end_logits
