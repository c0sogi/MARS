import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling: Learns a scalar weight for each of the last N
    hidden layers and computes their weighted average using the [CLS] token.
    """

    def __init__(self, num_pooling_layers: int = 4):
        super(WeightedLayerPooling, self).__init__()
        self.num_pooling_layers = num_pooling_layers

        # Initialize learnable weights for the layers (initialized to 1.0)
        self.layer_weights = nn.Parameter(
            torch.ones(num_pooling_layers, dtype=torch.float32)
        )

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states: Tuple of tensors from the backbone.
                               Shape of each: (batch_size, seq_len, hidden_dim)
        Returns:
            weighted_embedding: (batch_size, hidden_dim)
        """
        # Select the last N layers
        # Note: all_hidden_states includes embeddings + layers.
        # Using negative indexing safely grabs the last N encoder outputs.
        selected_layers = all_hidden_states[-self.num_pooling_layers :]

        # Extract [CLS] token (index 0) from each selected layer
        # Stack shape: (batch_size, num_pooling_layers, hidden_dim)
        cls_embeddings = torch.stack(
            [layer[:, 0, :] for layer in selected_layers], dim=1
        )

        # Compute softmax-normalized weights
        # weights shape: (num_pooling_layers,)
        weights = F.softmax(self.layer_weights, dim=0)

        # Reshape weights for broadcasting: (1, num_pooling_layers, 1)
        w = weights.view(1, self.num_pooling_layers, 1)

        # Compute weighted sum across the layer dimension
        # Output shape: (batch_size, hidden_dim)
        weighted_embedding = (w * cls_embeddings).sum(dim=1)

        return weighted_embedding


class SiameseDeberta(nn.Module):
    """
    Siamese DeBERTa-v3-Base model.
    Encodes two text inputs using a shared backbone, applies weighted layer pooling,
    computes interaction terms, injects scalar features, and predicts the winner.
    """

    def __init__(self):
        super(SiameseDeberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True

        # Load Backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Feature Dimensions
        self.hidden_size = self.config.hidden_size
        self.num_features = 8  # As defined in library.data.py (len_a, len_b, words, newlines, ratio, diff)

        # Pooling Mechanism
        if Config.USE_WEIGHTED_LAYER_POOLING:
            self.pooling = WeightedLayerPooling(
                num_pooling_layers=Config.POOLING_LAYERS
            )
        else:
            self.pooling = None

        # Classifier Head
        # Inputs:
        # 1. Embedding A (u)
        # 2. Embedding B (v)
        # 3. Absolute Difference (|u - v|)
        # 4. Element-wise Product (u * v)
        # 5. Scalar Features
        input_dim = (4 * self.hidden_size) + self.num_features

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, Config.NUM_CLASSES),
        )

        # Initialize weights for the custom head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize weights for the classifier head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_branch(self, input_ids, attention_mask):
        """
        Passes a single input branch through the backbone and pooling layer.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling is not None:
            # outputs.hidden_states contains all layer outputs
            pooled_output = self.pooling(outputs.hidden_states)
        else:
            # Fallback: Standard [CLS] pooling from the last hidden state
            # DeBERTa v3 uses the first token as [CLS]
            pooled_output = outputs.last_hidden_state[:, 0, :]

        return pooled_output

    def forward(
        self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, features
    ):
        """
        Forward pass for the Siamese network.

        Args:
            input_ids_a, attention_mask_a: Inputs for Model A response
            input_ids_b, attention_mask_b: Inputs for Model B response
            features: Explicit scalar features (batch, 8)
        """
        # 1. Encode Branch A
        u = self.forward_branch(input_ids_a, attention_mask_a)

        # 2. Encode Branch B
        v = self.forward_branch(input_ids_b, attention_mask_b)

        # 3. Compute Interaction Terms
        diff_abs = torch.abs(u - v)
        prod = u * v

        # 4. Concatenate All Signals
        # [u, v, |u-v|, u*v, features]
        combined = torch.cat([u, v, diff_abs, prod, features], dim=1)

        # 5. Classification
        logits = self.classifier(combined)

        return logits
