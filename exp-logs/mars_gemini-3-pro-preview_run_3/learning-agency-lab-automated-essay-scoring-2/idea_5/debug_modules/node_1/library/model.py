import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer that computes a weighted average of token embeddings.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # Calculate attention weights: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens by setting them to a very small value
        # attention_mask is 1 for valid tokens, 0 for padding
        w = w.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)

        # Apply softmax to get normalized weights
        w = torch.softmax(w, dim=1)

        # Weighted sum of hidden states: (batch_size, hidden_size)
        c = torch.sum(last_hidden_state * w, dim=1)
        return c


class EssayModel(nn.Module):
    """
    Main model class combining DeBERTa backbone, Attention Pooling, and a Regression Head.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load Backbone
        model_config = AutoConfig.from_pretrained(config.model_name)
        model_config.attention_probs_dropout_prob = 0.0
        model_config.hidden_dropout_prob = 0.0

        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=model_config
        )

        # Gradient Checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pool = AttentionPooling(model_config.hidden_size)

        # Regression Head
        self.fc = nn.Linear(model_config.hidden_size, 1)

        # Initialize weights for the head
        self._init_weights(self.pool)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.model_name == "microsoft/deberta-v3-large"
                and 0.02
                or 0.02,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        Returns:
            logits: (batch_size,) - The predicted continuous score.
        """
        # Backbone forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Pooling
        feature = self.pool(last_hidden_state, attention_mask)

        # Regression Head
        logits = self.fc(feature)

        # Squeeze to shape (batch_size,)
        return logits.squeeze(-1)


def get_optimizer_params(model, config: Config):
    """
    Constructs parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model: The EssayModel instance.
        config: Configuration object containing learning rates and decay factors.

    Returns:
        list: A list of dictionaries defining parameter groups.
    """
    # Base learning rate
    lr = config.learning_rate
    weight_decay = config.weight_decay
    llrd_decay = config.llrd_decay if config.use_llrd else 1.0

    # Define parameter groups
    optimizer_parameters = []

    # 1. Group: Regression Head and Pooling (Highest LR)
    # These layers are initialized randomly and need to learn the most
    head_params = list(model.pool.named_parameters()) + list(
        model.fc.named_parameters()
    )
    optimizer_parameters.append(
        {"params": [p for n, p in head_params], "lr": lr, "weight_decay": weight_decay}
    )

    # 2. Group: Backbone Layers (Decaying LR)
    # DeBERTa v3 structure: embeddings -> encoder.layer.0 ... encoder.layer.N

    # Get all named parameters of the backbone
    backbone_params = list(model.backbone.named_parameters())

    # Identify layers
    # Embeddings
    embeddings_params = []
    # Encoder layers: stored in a dict keyed by layer index
    layers_params = {}
    # Other backbone params (like final layernorm in encoder if exists, or pooler if exists)
    others_params = []

    # DeBERTa specific naming convention check
    # usually: backbone.embeddings... and backbone.encoder.layer.X...

    # Determine number of layers automatically
    if hasattr(model.backbone.config, "num_hidden_layers"):
        num_layers = model.backbone.config.num_hidden_layers
    else:
        # Fallback for standard BERT-likes
        num_layers = 24  # for large

    for name, param in backbone_params:
        if "embeddings" in name:
            embeddings_params.append(param)
        elif "encoder.layer" in name:
            # Extract layer index
            # Example: encoder.layer.11.output.dense.weight
            try:
                # split by dot, find 'layer', next is index
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layer") + 1])
                if layer_idx not in layers_params:
                    layers_params[layer_idx] = []
                layers_params[layer_idx].append(param)
            except (ValueError, IndexError):
                others_params.append(param)
        else:
            others_params.append(param)

    # Assign LRs to Encoder Layers
    # Layer N-1 (top) gets lr * (decay^0)
    # Layer 0 (bottom) gets lr * (decay^(N-1))
    for layer_idx in range(num_layers - 1, -1, -1):
        decay_power = num_layers - 1 - layer_idx + 1  # +1 because head is 0 decay
        layer_lr = lr * (llrd_decay**decay_power)

        if layer_idx in layers_params:
            optimizer_parameters.append(
                {
                    "params": layers_params[layer_idx],
                    "lr": layer_lr,
                    "weight_decay": weight_decay,
                }
            )

    # Assign LR to Embeddings
    # Embeddings are below layer 0
    embeddings_lr = lr * (llrd_decay ** (num_layers + 1))
    if embeddings_params:
        optimizer_parameters.append(
            {
                "params": embeddings_params,
                "lr": embeddings_lr,
                "weight_decay": weight_decay,
            }
        )

    # Assign LR to any other backbone parameters (e.g. relative attention bias usually in encoder but not in layer)
    # We treat them similar to embeddings or the lowest layer
    if others_params:
        optimizer_parameters.append(
            {"params": others_params, "lr": embeddings_lr, "weight_decay": weight_decay}
        )

    return optimizer_parameters
