import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from library.config import CFG


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling (WLP) aggregates the [CLS] token embeddings from the
    last `num_pooling_layers` of the backbone using learnable weights.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor(
                    [1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
                )
            )
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of embeddings from all layers
        # We select the layers we want to pool (e.g., the last 4)
        # The tuple includes the initial embedding layer, so indices might need adjustment
        # usually all_hidden_states[-1] is the last layer.

        # Extract the [CLS] token (index 0) from the specified last N layers
        all_layer_embedding = torch.stack(
            [
                all_hidden_states[i][:, 0, :]
                for i in range(self.layer_start, self.num_hidden_layers + 1)
            ],
            dim=-1,
        )

        # Apply softmax to weights to ensure they sum to 1
        weight = F.softmax(self.layer_weights, dim=0)

        # Weighted sum: (Batch, Hidden, Layers) * (Layers) -> (Batch, Hidden)
        # We broadcast the weights across the batch and hidden dimensions
        outputs = torch.sum(all_layer_embedding * weight, dim=-1)

        return outputs


class CustomModel(nn.Module):
    """
    Custom DeBERTa-v3 model with Weighted Layer Pooling and Multi-Sample Dropout.
    """

    def __init__(self, cfg=CFG, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg

        # Load Configuration
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model_name, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
            self.config.output_hidden_states = True

        # Load Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Gradient Checkpointing
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Weighted Layer Pooling
        # We pool the last N layers.
        # DeBERTa-Large has 24 layers. hidden_states tuple size is 25 (embeddings + 24 layers).
        # To pool last 4: indices 21, 22, 23, 24.
        # layer_start = total_layers - num_pooling + 1?
        # If total is 24, and we want last 4 (21, 22, 23, 24), start is 21.
        # 24 - 4 + 1 = 21.
        # Note: config.num_hidden_layers is 24.

        self.num_pooling_layers = cfg.num_pooling_layers
        layer_start = self.config.num_hidden_layers - self.num_pooling_layers + 1
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers, layer_start=layer_start
        )

        # Multi-Sample Dropout (MSD) Head
        self.dropouts = nn.ModuleList(
            [nn.Dropout(cfg.fc_dropout) for _ in range(cfg.num_msd)]
        )

        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Backbone Forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get hidden states from all layers
        all_hidden_states = outputs.hidden_states

        # Apply Weighted Layer Pooling
        feature = self.pooler(all_hidden_states)

        # Multi-Sample Dropout Head
        # Apply multiple dropouts and average the predictions
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.fc(dropout(feature))
            else:
                output += self.fc(dropout(feature))

        output /= len(self.dropouts)

        # Flatten to shape (Batch,)
        return output.squeeze(-1)


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures optimizer parameters with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model: The CustomModel instance.
        encoder_lr: Base learning rate for the transformer encoder.
        decoder_lr: Learning rate for the custom head.
        weight_decay: Weight decay coefficient.

    Returns:
        List of dictionaries defining parameter groups.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []

    # LLRD Configuration
    # DeBERTa-v3-large has 24 layers.
    # We assign learning rates: lr * (decay ^ depth)
    # Head gets decoder_lr.
    # Top layer gets encoder_lr.
    # Bottom layers get decayed encoder_lr.

    num_layers = model.config.num_hidden_layers
    llrd_decay = CFG.llrd_decay

    # 1. Group Parameters by Layer
    # We create buckets for: embeddings, layer.0 ... layer.23, and the head (pooler+fc)

    # Initialize groups
    # Group 0: Embeddings (Depth = num_layers + 1 for decay calculation purposes, i.e., furthest)
    # Group 1..N: Encoder Layers (Depth = num_layers - layer_idx)
    # Group N+1: Head (Depth = 0)

    # Helper to determine layer index from parameter name
    def get_layer_index(name):
        if "embeddings" in name:
            return -1  # Embeddings
        if "encoder.layer" in name:
            # Format: model.encoder.layer.X. ...
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layer":
                    return int(parts[i + 1])
        return None  # Head or other

    for name, p in param_optimizer:
        if not p.requires_grad:
            continue

        layer_idx = get_layer_index(name)

        # Determine Learning Rate
        if layer_idx is None:
            # Head parameters (pooler, fc, etc.)
            lr = decoder_lr
        elif layer_idx == -1:
            # Embeddings: furthest away
            lr = encoder_lr * (llrd_decay ** (num_layers + 1))
        else:
            # Encoder Layers: layer 23 is closest to head (decay^0 = 1), layer 0 is furthest
            # Decay factor: num_layers - 1 - layer_idx
            # e.g. layer 23 -> 24 - 1 - 23 = 0 -> lr * 1.0
            # e.g. layer 0  -> 24 - 1 - 0  = 23 -> lr * decay^23
            lr = encoder_lr * (llrd_decay ** (num_layers - 1 - layer_idx))

        # Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        optimizer_parameters.append({"params": [p], "weight_decay": wd, "lr": lr})

    return optimizer_parameters
