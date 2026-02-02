import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling strategy.
    Learns a weighted average of the last `num_hidden_layers` from the transformer backbone.
    """

    def __init__(self, num_hidden_layers: int = 4):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights to be equal; softmax will be applied in forward pass
        self.layer_weights = nn.Parameter(
            torch.tensor([1.0] * num_hidden_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states: Tuple of tensor (batch, seq_len, hidden_dim).
                               Contains embedding output + output of each layer.
        Returns:
            torch.Tensor: Pooled representation (batch, hidden_dim)
        """
        # Select the last N layers
        # all_hidden_states contains (embeddings, layer_1, ..., layer_N)
        selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack layers to shape: (batch, seq_len, hidden_dim, num_pooling_layers)
        stacked_layers = torch.stack(selected_layers, dim=-1)

        # Compute normalized weights
        # weights shape: (num_pooling_layers,)
        weights = torch.softmax(self.layer_weights, dim=0)

        # Reshape weights for broadcasting: (1, 1, 1, num_pooling_layers)
        weights = weights.view(1, 1, 1, -1)

        # Weighted sum across the last dimension
        # Result shape: (batch, seq_len, hidden_dim)
        weighted_sum = (stacked_layers * weights).sum(dim=-1)

        # Extract the representation of the first token (CLS)
        # Result shape: (batch, hidden_dim)
        cls_output = weighted_sum[:, 0, :]

        return cls_output


class DebertaClassifier(nn.Module):
    """
    DeBERTa-v3-Large based classifier with Weighted Layer Pooling.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=3):
        super(DebertaClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        if Config.USE_WEIGHTED_LAYER_POOLING:
            self.pooling = WeightedLayerPooling(
                num_hidden_layers=Config.NUM_POOLING_LAYERS
            )
        else:
            self.pooling = None

        self.fc = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize the classification head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # Pooling
        if self.pooling:
            cls_embedding = self.pooling(all_hidden_states)
        else:
            # Fallback to standard last layer CLS token if pooling is disabled
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Classification
        logits = self.fc(cls_embedding)

        return logits


def get_llrd_optimizer_params(
    model,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    decay_factor=Config.LLRD_DECAY,
):
    """
    Generates optimizer parameters for Layer-Wise Learning Rate Decay (LLRD).

    Args:
        model (nn.Module): The model to optimize.
        learning_rate (float): Base learning rate for the head/top layers.
        weight_decay (float): Weight decay for regularization.
        decay_factor (float): Multiplicative decay factor for lower layers.

    Returns:
        list: List of parameter groups for the optimizer.
    """
    # Determine the number of layers in the backbone
    # DeBERTa v3 large usually has 24 layers
    num_layers = model.config.num_hidden_layers

    # Dictionary to group parameters by (lr, weight_decay) to avoid many small groups
    param_groups = {}

    named_parameters = list(model.named_parameters())

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        # 1. Determine Layer ID
        layer_id = None

        # Check for embeddings
        if "backbone.embeddings" in name or "backbone.encoder.rel_embeddings" in name:
            layer_id = 0
        elif "backbone.encoder.layer." in name:
            # Extract layer index from name like 'backbone.encoder.layer.15.output...'
            parts = name.split(".")
            try:
                # Find the part after 'layer'
                idx = parts.index("layer")
                layer_id = int(parts[idx + 1]) + 1  # +1 because embeddings is level 0
            except (ValueError, IndexError):
                layer_id = 0  # Fallback to lowest LR if parsing fails
        else:
            # This covers the task head (fc) and pooling layer weights
            # Assign them to the "top" level
            layer_id = num_layers + 1

        # 2. Calculate Learning Rate
        # Formula: lr = base_lr * (decay_factor ^ (max_layer - current_layer))
        # Head (layer_id = num_layers + 1) gets base_lr
        # Top Transformer Layer (layer_id = num_layers) gets base_lr * decay
        exponent = (num_layers + 1) - layer_id
        current_lr = learning_rate * (decay_factor**exponent)

        # 3. Determine Weight Decay
        # Exclude bias and LayerNorm weights from weight decay
        if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
            current_wd = 0.0
        else:
            current_wd = weight_decay

        # 4. Group Parameters
        key = (current_lr, current_wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    # Convert to list of dicts required by PyTorch optimizers
    optimizer_grouped_parameters = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in param_groups.items()
    ]

    return optimizer_grouped_parameters
