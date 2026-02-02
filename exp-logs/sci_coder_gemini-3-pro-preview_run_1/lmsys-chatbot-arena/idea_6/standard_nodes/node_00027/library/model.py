import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted sum of the last N hidden layers.
    Returns a sequence representation (Batch, SeqLen, Hidden).
    """

    def __init__(self, num_hidden_layers=12, layer_start=8, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.num_layers_to_mix = (
            num_hidden_layers - layer_start + 1
        )  # +1 includes the final embedding

        # Learnable weights for the layers
        self.layer_weights = nn.Parameter(torch.tensor([1.0] * self.num_layers_to_mix))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (initial_embeds, layer_1, ..., layer_12)
        # We are interested in the last N layers.
        # Note: AutoModel output.hidden_states includes the initial embeddings at index 0.
        # So layer 1 is at index 1.

        # Select the layers we want to mix
        # If layer_start is 9 (1-based from config usually means last 4 layers of 12 -> 9,10,11,12)
        # In 0-indexing tuple: indices 9, 10, 11, 12.

        # Config.USE_LAST_N_LAYERS = 4.
        # Total layers in base = 12. Output tuple size = 13.
        # We want indices [-4:].

        selected_layers = all_hidden_states[-Config.USE_LAST_N_LAYERS :]

        # Stack: (Batch, Seq, Hidden, NumLayers)
        stacked_layers = torch.stack(selected_layers, dim=-1)

        # Normalize weights
        weights = F.softmax(self.layer_weights, dim=0)

        # Weighted sum: (Batch, Seq, Hidden)
        # sum( H_i * w_i )
        weighted_sum = (stacked_layers * weights.view(1, 1, 1, -1)).sum(dim=-1)

        return weighted_sum


class SiameseDebertaGeometric(nn.Module):
    """
    Siamese DeBERTa-v3-Base with Weighted Layer Mixing and Geometric Interaction.
    Replaces Cross-Attention with explicit geometric primitives (|u-v|, u*v)
    for better inductive bias in ranking tasks.
    """

    def __init__(self):
        super(SiameseDebertaGeometric, self).__init__()

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # 2. Weighted Layer Pooling
        layer_start = self.config.num_hidden_layers - Config.USE_LAST_N_LAYERS + 1
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers, layer_start=layer_start
        )

        # 3. Classifier Head
        # Input: [u, v, |u-v|, u*v, Scalars]
        self.scalar_dim = 6
        self.fusion_dim = (Config.HIDDEN_SIZE * 4) + self.scalar_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, Config.HIDDEN_SIZE),
            nn.LayerNorm(Config.HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_PROB),
            nn.Linear(Config.HIDDEN_SIZE, Config.NUM_CLASSES),
        )

        # Initialize weights for new layers
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.ModuleList):
            for submodule in module:
                self._init_weights(submodule)

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalar_features,
    ):
        # --- 1. Siamese Encoding ---
        out_a = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        out_b = self.backbone(input_ids=input_ids_b, attention_mask=attention_mask_b)

        # --- 2. Weighted Layer Mixing ---
        seq_a = self.pooler(out_a.hidden_states)
        seq_b = self.pooler(out_b.hidden_states)

        # --- 3. Mean Pooling ---
        def mean_pooling(hidden_states, attention_mask):
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

        u = mean_pooling(seq_a, attention_mask_a)
        v = mean_pooling(seq_b, attention_mask_b)

        # --- 4. Geometric Interaction ---
        # Cite solution_lesson_node_00025: Explicit geometric interaction terms
        # provide superior inductive bias compared to learnable attention.
        diff = torch.abs(u - v)
        prod = u * v

        # --- 5. Fusion & Classification ---
        # Concatenate: [u, v, |u-v|, u*v, Scalars]
        combined = torch.cat([u, v, diff, prod, scalar_features], dim=1)

        logits = self.classifier(combined)

        return logits
