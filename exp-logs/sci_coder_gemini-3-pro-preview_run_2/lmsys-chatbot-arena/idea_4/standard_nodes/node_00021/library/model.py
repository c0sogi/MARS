import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Applies a weighted average over the last 'layer_start' hidden layers
    of the transformer backbone, followed by Mean Pooling.
    The weights for the layers are learnable.
    """

    def __init__(self, num_hidden_layers, layer_start=4, hidden_size=768):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        # Learnable weights for the selected layers
        self.layer_weights = nn.Parameter(
            torch.tensor([1.0] * layer_start, dtype=torch.float)
        )

    def forward(self, all_hidden_states, attention_mask):
        """
        Args:
            all_hidden_states: Tuple of tensors (batch, seq_len, hidden) from backbone.
            attention_mask: Tensor (batch, seq_len) indicating valid tokens.

        Returns:
            Tensor (batch, hidden_size) representing the pooled embedding.
        """
        # Select the last 'layer_start' layers
        # all_hidden_states contains (embeddings, layer_1, ..., layer_N)
        # We take the last N layers.
        all_layer_embeddings = all_hidden_states[-self.layer_start :]

        # Stack to shape: (batch, layer_start, seq_len, hidden)
        all_layer_embeddings = torch.stack(all_layer_embeddings, dim=1)

        # Compute softmax normalized weights
        # shape: (layer_start) -> (1, layer_start, 1, 1) for broadcasting
        weight_factor = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)

        # Weighted sum across the layer dimension
        # Result shape: (batch, seq_len, hidden)
        weighted_average = (all_layer_embeddings * weight_factor).sum(dim=1)

        # Mean Pooling with Attention Mask
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(weighted_average.size()).float()
        )

        # Sum embeddings of non-padding tokens
        sum_embeddings = torch.sum(weighted_average * input_mask_expanded, 1)

        # Count non-padding tokens (clamp to avoid division by zero)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask


class SiameseDeberta(nn.Module):
    """
    Siamese Network using DeBERTa-v3-base backbone.
    Processes two branches (A and B) and combines them with meta-features
    for preference prediction.
    """

    def __init__(self):
        super(SiameseDeberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True

        # Initialize Backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Initialize Pooling Layer
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_start=4,
            hidden_size=self.config.hidden_size,
        )

        # Feature Dimensions
        # u, v, |u-v|, u*v -> 4 vectors
        embedding_dim = 4 * self.config.hidden_size
        meta_dim = 3  # Prompt len, Res A len, Res B len
        input_dim = embedding_dim + meta_dim

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.config.hidden_size, Config.NUM_LABELS),
        )

        # Initialize weights for the custom head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_one_branch(self, input_ids, attention_mask):
        """
        Passes one branch (Prompt + Response) through backbone and pooler.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.hidden_states is a tuple of all layer outputs
        pooled_output = self.pooler(outputs.hidden_states, attention_mask)
        return pooled_output

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        meta_features,
    ):
        """
        Forward pass for the Siamese network.

        Args:
            input_ids_a, attention_mask_a: Inputs for Branch A
            input_ids_b, attention_mask_b: Inputs for Branch B
            meta_features: Normalized length features (batch, 3)

        Returns:
            logits: (batch, 3)
        """
        # 1. Get Embeddings for both branches
        u = self.forward_one_branch(input_ids_a, attention_mask_a)
        v = self.forward_one_branch(input_ids_b, attention_mask_b)

        # 2. Compute Interaction Features
        diff_uv = torch.abs(u - v)
        prod_uv = u * v

        # 3. Concatenate all features
        # [u, v, |u-v|, u*v, meta_features]
        features = torch.cat([u, v, diff_uv, prod_uv, meta_features], dim=1)

        # 4. Classification
        logits = self.classifier(features)

        return logits
