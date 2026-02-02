import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Pools the last N layers of the transformer encoder using learnable weights.
    """

    def __init__(
        self, num_hidden_layers: int, layer_start: int = 4, hidden_size: int = 768
    ):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_pooling_layers = Config.NUM_POOLING_LAYERS

        # Learnable weights for the last n layers
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * self.num_pooling_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensor (batch, seq_len, hidden_size)
        # We take the last num_pooling_layers
        all_layer_embeddings = all_hidden_states[-self.num_pooling_layers :]

        # Stack to (num_pooling_layers, batch, seq_len, hidden_size)
        all_layer_embeddings = torch.stack(all_layer_embeddings)

        # Compute softmax weights
        weight_factor = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum: (batch, seq_len, hidden_size)
        weighted_average = (weight_factor * all_layer_embeddings).sum(dim=0)

        # We use the embedding of the [CLS] token (index 0) as the sequence representation
        # Shape: (batch, hidden_size)
        return weighted_average[:, 0]


class SiameseDebertaGated(nn.Module):
    """
    Siamese DeBERTa-v3-Base with Gated Semantic-Structural Fusion.
    """

    def __init__(self):
        super(SiameseDebertaGated, self).__init__()

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # 2. Pooling
        self.pooling = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_start=self.config.num_hidden_layers - Config.NUM_POOLING_LAYERS,
            hidden_size=Config.HIDDEN_SIZE,
        )

        # 3. Feature Dimensions
        # Semantic Stream: [u, v, |u-v|, u*v] -> 4 * Hidden Size
        self.semantic_input_dim = 4 * Config.HIDDEN_SIZE

        # Structural Stream: 6 features (char_diff, char_ratio, word_diff, word_ratio, newline_diff, newline_ratio)
        self.structural_input_dim = 6

        # Shared Latent Dimension for Fusion
        self.latent_dim = Config.HIDDEN_SIZE

        # 4. Projections
        # Project Semantic Features to Latent Space
        self.semantic_projection = nn.Sequential(
            nn.Linear(self.semantic_input_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # Project Structural Features to Latent Space
        self.structural_projection = nn.Sequential(
            nn.Linear(self.structural_input_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # 5. Gating Mechanism
        # Input: Concatenation of projected semantic and structural features
        # Output: Scalar alpha (0 to 1)
        self.gate_net = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 6. Classifier
        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, Config.NUM_LABELS))

        # Weight Initialization for new layers
        self._init_weights(self.semantic_projection)
        self._init_weights(self.structural_projection)
        self._init_weights(self.gate_net)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_encoder(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.hidden_states is a tuple of (batch, seq_len, hidden_size)
        pooled_output = self.pooling(outputs.hidden_states)
        return pooled_output

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        structural_features,
    ):
        """
        Args:
            input_ids_a, attention_mask_a: Inputs for Prompt + Response A
            input_ids_b, attention_mask_b: Inputs for Prompt + Response B
            structural_features: Tensor of shape (Batch, 6)
        """
        # 1. Siamese Encoding
        u = self.forward_encoder(input_ids_a, attention_mask_a)  # (Batch, Hidden)
        v = self.forward_encoder(input_ids_b, attention_mask_b)  # (Batch, Hidden)

        # 2. Semantic Stream Construction
        # Features: u, v, |u-v|, u*v
        diff_abs = torch.abs(u - v)
        prod = u * v
        semantic_raw = torch.cat([u, v, diff_abs, prod], dim=1)  # (Batch, 4*Hidden)

        # 3. Projections
        h_sem = self.semantic_projection(semantic_raw)  # (Batch, Latent)
        h_struct = self.structural_projection(structural_features)  # (Batch, Latent)

        # 4. Gated Fusion
        # Concatenate for gate input
        gate_input = torch.cat([h_sem, h_struct], dim=1)  # (Batch, 2*Latent)
        alpha = self.gate_net(gate_input)  # (Batch, 1)

        # Weighted Combination
        # alpha * Semantic + (1 - alpha) * Structural
        h_fused = alpha * h_sem + (1 - alpha) * h_struct  # (Batch, Latent)

        # 5. Classification
        logits = self.classifier(h_fused)

        return logits
