import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling strategy.
    Computes a learnable weighted average of the last `n` hidden layers,
    followed by mean pooling over the sequence dimension.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 4, hidden_size: int = 768):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_layers_to_pool = self.num_hidden_layers - self.layer_start

        # Learnable weights for the layers
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * (num_hidden_layers - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states, attention_mask):
        # all_hidden_states is a tuple of (batch, seq_len, hidden_size)
        # We take the last few layers
        all_layer_embedding = torch.stack(
            list(all_hidden_states)[self.layer_start :], dim=0
        )

        # Compute softmax over the layer weights
        weight_factor = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum of layers: (num_layers, batch, seq, hidden) -> (batch, seq, hidden)
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0)

        # Mean Pooling over the sequence dimension, ignoring padding
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(weighted_average.size()).float()
        )
        sum_embeddings = torch.sum(weighted_average * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask


class SiameseDeberta(nn.Module):
    """
    Siamese DeBERTa-v3-Base model with Weighted Layer Pooling and geometric interaction features.
    """

    def __init__(self):
        super(SiameseDeberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.output_hidden_states = True
        self.config.hidden_dropout_prob = Config.hidden_dropout_prob
        self.config.attention_probs_dropout_prob = Config.attention_probs_dropout_prob

        # Backbone
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Enable Gradient Checkpointing for memory efficiency if needed
        # self.backbone.gradient_checkpointing_enable()

        # Pooling
        # We determine the start layer index based on total layers and n_last_layers
        total_layers = (
            self.config.num_hidden_layers + 1
        )  # +1 for embedding layer if included, usually num_hidden_layers is enough for encoder layers
        # DeBERTa output_hidden_states includes the initial embeddings + 12 encoder layers = 13 tensors
        # We want the last N encoder layers.
        # If we use negative indexing logic in the pooling class, it's cleaner.
        # Let's adjust WeightedLayerPooling to handle negative indexing or explicit slicing logic.
        # Here we pass the explicit start index.
        # total tensors returned = 13 (0..12). We want last 4: 9, 10, 11, 12.
        # Start index = 13 - 4 = 9.
        total_output_tensors = self.config.num_hidden_layers + 1
        start_index = total_output_tensors - Config.n_last_layers

        self.pooling = WeightedLayerPooling(
            num_hidden_layers=total_output_tensors,
            layer_start=start_index,
            hidden_size=self.config.hidden_size,
        )

        # Classification Head
        # Input: u (hidden), v (hidden), |u-v| (hidden), u*v (hidden), scalars (6)
        # Total dim = 4 * hidden_size + num_scalar_features
        input_dim = (4 * self.config.hidden_size) + Config.num_scalar_features

        self.head = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, 3),
        )

        # Initialize weights for the head
        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_one_branch(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.hidden_states is a tuple of tensors
        pooled_output = self.pooling(outputs.hidden_states, attention_mask)
        return pooled_output

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalar_features,
    ):
        # Encode Response A
        u = self.forward_one_branch(input_ids_a, attention_mask_a)

        # Encode Response B
        v = self.forward_one_branch(input_ids_b, attention_mask_b)

        # Geometric Interaction Terms
        abs_diff = torch.abs(u - v)
        prod = u * v

        # Concatenate: [u, v, |u-v|, u*v, scalars]
        # scalar_features shape: (batch, 6)
        features = torch.cat([u, v, abs_diff, prod, scalar_features], dim=1)

        # Classification
        logits = self.head(features)

        return logits
