import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling: Learns a scalar weight for each layer's output
    and computes a weighted average of the hidden states.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 1, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers

        # Initialize weights for the layers we want to pool
        # We add 1 to num_hidden_layers because hidden_states includes embeddings
        num_layers_to_pool = num_hidden_layers + 1 - layer_start

        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor([1.0] * num_layers_to_pool, dtype=torch.float)
            )
        )

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states (tuple): Tuple of tensors containing hidden states from the backbone.
                                       Shape of each: (Batch, Seq, Dim)
        Returns:
            torch.Tensor: Weighted average of hidden states. Shape: (Batch, Seq, Dim)
        """
        # Stack hidden states: (Num_Layers, Batch, Seq, Dim)
        all_layer_embedding = torch.stack(list(all_hidden_states), dim=0)

        # Select layers starting from layer_start (e.g., skip embeddings at index 0)
        all_layer_embedding = all_layer_embedding[self.layer_start :, :, :, :]

        # Compute softmax of learnable weights to ensure they sum to 1
        weight_factor = torch.nn.functional.softmax(self.layer_weights, dim=0)

        # Perform weighted sum
        # Reshape weights for broadcasting: (Num_Layers, 1, 1, 1)
        weighted_average = (weight_factor.view(-1, 1, 1, 1) * all_layer_embedding).sum(
            dim=0
        )

        return weighted_average


class AttentionPooling(nn.Module):
    """
    Attention Pooling: Computes a weighted average of token embeddings based on their
    relevance (attention scores).
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state (torch.Tensor): Input features (Batch, Seq, Dim)
            attention_mask (torch.Tensor): Attention mask (Batch, Seq)
        Returns:
            torch.Tensor: Pooled feature vector (Batch, Dim)
        """
        # Compute raw attention scores: (Batch, Seq, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens so they don't contribute to the pool
        # Using -1e4 instead of -1e9 to avoid overflow in FP16 (min ~ -65504)
        w = w.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)

        # Normalize scores
        w = torch.softmax(w, dim=1)

        # Weighted sum of token embeddings
        feature = torch.sum(w * last_hidden_state, dim=1)
        return feature


class CustomModel(nn.Module):
    """
    Custom Model for Phrase Matching.
    Backbone: DeBERTa-v3-Large
    Head: Weighted Layer Pooling -> Attention Pooling -> Multi-Sample Dropout -> Dual Heads
    """

    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = Config
        model_name = config_path if config_path else self.cfg.model_name

        # Load Backbone Configuration
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        # Load Backbone Model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # 1. Weighted Layer Pooling
        if self.cfg.use_weighted_layer_pooling:
            self.pooler = WeightedLayerPooling(
                num_hidden_layers=self.config.num_hidden_layers,
                layer_start=1,  # Skip embedding layer
            )

        # 2. Attention Pooling
        if self.cfg.use_attention_pooling:
            self.attention_pooler = AttentionPooling(self.config.hidden_size)

        # 3. Output Heads
        # Regression Head
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.num_classes)
        # Auxiliary Classification Head
        self.fc_class = nn.Linear(self.config.hidden_size, self.cfg.aux_num_classes)

        # Initialize custom weights
        self._init_weights(self.fc)
        self._init_weights(self.fc_class)
        if self.cfg.use_attention_pooling:
            for module in self.attention_pooler.modules():
                self._init_weights(module)

    def _init_weights(self, module):
        """
        Initialize weights for custom modules using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        Returns dictionary with 'logits' (regression) and 'class_logits' (classification).
        """
        # Pass through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 1. Apply Weighted Layer Pooling
        if self.cfg.use_weighted_layer_pooling:
            feature = self.pooler(all_hidden_states)
        else:
            feature = outputs.last_hidden_state

        # 2. Apply Attention Pooling
        if self.cfg.use_attention_pooling:
            feature = self.attention_pooler(feature, attention_mask)
        else:
            # Fallback: Mean Pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(feature.size()).float()
            )
            sum_embeddings = torch.sum(feature * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            feature = sum_embeddings / sum_mask

        # 3. Multi-Sample Dropout & Heads
        logits_list = []
        class_logits_list = []

        if self.cfg.use_multi_sample_dropout:
            # Apply multiple dropout masks and average the predictions
            for rate in self.cfg.multi_sample_dropout_rates:
                x = torch.nn.functional.dropout(feature, p=rate, training=self.training)
                logits_list.append(self.fc(x))
                class_logits_list.append(self.fc_class(x))

            # Average predictions
            logits = torch.mean(torch.stack(logits_list), dim=0)
            class_logits = torch.mean(torch.stack(class_logits_list), dim=0)
        else:
            # Standard single dropout
            x = torch.nn.functional.dropout(
                feature, p=self.cfg.dropout_rate, training=self.training
            )
            logits = self.fc(x)
            class_logits = self.fc_class(x)

        return {"logits": logits, "class_logits": class_logits}
