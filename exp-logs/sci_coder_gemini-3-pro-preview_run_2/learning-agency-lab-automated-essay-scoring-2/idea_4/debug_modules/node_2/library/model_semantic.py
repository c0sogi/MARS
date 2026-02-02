import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class DebertaDualPool(nn.Module):
    """
    DeBERTa-v3-Large with Dual Pooling (Mean + Max) and a Linear Head.

    Architecture:
    1. Backbone: microsoft/deberta-v3-large
    2. Pooling: Concatenation of Mean Pooling and Max Pooling
    3. Head: Linear layer projecting (Hidden * 2) -> 1
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)

        # Enable gradient checkpointing to save VRAM with Large model
        # This is critical for A100 40GB when using max_len=1024
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Dual pooling output dimension = hidden_size * 2
        self.pooling_output_dim = self.config.hidden_size * 2
        self.fc = nn.Linear(self.pooling_output_dim, 1)

        # Initialize the linear head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize the weights of the linear head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Backbone forward pass
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Expand attention mask for broadcasting
        # attention_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # --- Mean Pooling ---
        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # Sum mask to get count of tokens
        sum_mask = input_mask_expanded.sum(1)
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # --- Max Pooling ---
        # Set padding tokens to a very small value so they aren't selected by max
        # Clone to avoid in-place modification errors in backward pass
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(embeddings, 1)[0]

        # --- Concatenation ---
        # [batch_size, hidden_size * 2]
        concat_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        # --- Projection ---
        # [batch_size, 1]
        logits = self.fc(concat_embeddings)

        return logits


def build_model():
    """
    Constructs and returns the DebertaDualPool model.
    """
    model = DebertaDualPool()
    return model


def get_optimizer(model):
    """
    Returns the AdamW optimizer configured as per the strategy (Uniform LR).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    return optimizer


def get_loss_fn():
    """
    Returns the SmoothL1Loss function.
    """
    return nn.SmoothL1Loss()
