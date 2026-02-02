import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Dynamic Layer Aggregation (Scalar Mixing).
    Learns a scalar weight for each layer of the transformer backbone to compute
    a weighted average of the hidden states.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 1, layer_weights=None):
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
        # Stack hidden states: (num_layers, batch_size, seq_len, hidden_dim)
        # We slice from layer_start to the end.
        # all_hidden_states usually includes embeddings as index 0, so total len is num_layers + 1
        all_layer_embedding = torch.stack(
            list(all_hidden_states)[self.layer_start :], dim=0
        )

        # Compute softmax weights
        weight_factor = torch.softmax(self.layer_weights, dim=0)

        # Weighted sum: (batch_size, seq_len, hidden_dim)
        # Broadcasting weights: (num_layers, 1, 1, 1) * (num_layers, B, S, H)
        weighted_average = (
            weight_factor[:, None, None, None] * all_layer_embedding
        ).sum(dim=0)

        return weighted_average


class AttentionPooling(nn.Module):
    """
    Attention-based Pooling layer.
    Computes a weighted average of token embeddings based on their importance.
    """

    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, hidden_dim)
        # attention_mask: (batch, seq_len)

        # Compute attention scores: (batch, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens (set to -inf)
        # attention_mask is 1 for tokens, 0 for padding
        # We want to set 0s to -1e9
        w[attention_mask == 0] = float("-inf")

        # Softmax over sequence length
        weights = torch.softmax(w, dim=1)

        # Weighted sum: (batch, hidden_dim)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)

        return context_vector


class PhraseModel(nn.Module):
    """
    Main model class for Phrase Matching.
    Features:
    - DeBERTa v3 Large Backbone
    - Weighted Layer Pooling
    - Attention Pooling
    - Multi-Sample Dropout
    - Dual Heads (Regression + Classification)
    """

    def __init__(self):
        super(PhraseModel, self).__init__()
        self.config = Config

        # Load Backbone Configuration
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.output_hidden_states = True

        # Initialize Backbone
        self.backbone = AutoModel.from_pretrained(
            self.config.model_name, config=model_config
        )

        # Feature Aggregation
        if self.config.use_weighted_layer_pooling:
            self.layer_pooler = WeightedLayerPooling(
                num_hidden_layers=model_config.num_hidden_layers,
                layer_start=1,  # Skip embeddings layer
            )

        # Pooling
        self.pooler = AttentionPooling(self.config.hidden_dim)

        # Multi-Sample Dropout
        # We use a loop in forward, so we just define one dropout layer here
        self.dropout = nn.Dropout(0.1)  # Standard dropout rate for MSD usually 0.1-0.5
        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(0.1) for _ in range(5)]  # 5 samples
        )

        # Heads
        self.fc = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim)
        self.activation = nn.GELU()

        # Regression Head (Score)
        self.regressor = nn.Linear(self.config.hidden_dim, 1)

        # Classification Head (Auxiliary Bins)
        self.classifier = nn.Linear(
            self.config.hidden_dim, self.config.num_classification_bins
        )

        # Initialize weights for new layers
        self._init_weights(self.fc)
        self._init_weights(self.regressor)
        self._init_weights(self.classifier)
        self._init_weights(self.layer_pooler.layer_weights)

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
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Backbone Forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 1. Aggregate Layers
        if self.config.use_weighted_layer_pooling:
            # outputs.hidden_states is a tuple of (embeddings, layer1, ..., layerN)
            sequence_output = self.layer_pooler(outputs.hidden_states)
        else:
            sequence_output = outputs.last_hidden_state

        # 2. Pooling
        features = self.pooler(sequence_output, attention_mask)

        # 3. Multi-Sample Dropout & Heads
        # Apply fully connected layer first (common shared representation)
        features = self.fc(features)
        features = self.layer_norm(features)
        features = self.activation(features)

        reg_logits_list = []
        class_logits_list = []

        for dropout_op in self.dropout_ops:
            dropped_features = dropout_op(features)

            # Regression
            reg_out = self.regressor(dropped_features)
            reg_logits_list.append(reg_out)

            # Classification
            class_out = self.classifier(dropped_features)
            class_logits_list.append(class_out)

        # Average predictions
        reg_logits = torch.mean(torch.stack(reg_logits_list), dim=0)
        class_logits = torch.mean(torch.stack(class_logits_list), dim=0)

        return {"logits": reg_logits, "class_logits": class_logits}

    def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay=0.0):
        """
        Layer-wise Learning Rate Decay (LLRD) implementation.
        Groups parameters and assigns decaying learning rates from top to bottom.
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        optimizer_parameters = []

        # Base LLRD configuration
        num_layers = self.backbone.config.num_hidden_layers
        decay_rate = self.config.llrd_decay

        # Initialize layer-wise learning rates
        # Layer 0 is bottom, Layer N is top.
        # We want Layer N to have encoder_lr, Layer N-1 to have encoder_lr * decay, etc.

        # 1. Backbone Layers
        for layer_idx in range(num_layers):
            # Calculate LR for this layer: lr * (decay ^ (num_layers - layer_idx))
            # Higher layer_idx (closer to output) -> Higher LR
            layer_lr = encoder_lr * (decay_rate ** (num_layers - 1 - layer_idx))

            layer_params = []
            for n, p in self.backbone.encoder.layer[layer_idx].named_parameters():
                layer_params.append((n, p))

            group_decay = [
                p for n, p in layer_params if not any(nd in n for nd in no_decay)
            ]
            group_no_decay = [
                p for n, p in layer_params if any(nd in n for nd in no_decay)
            ]

            if group_decay:
                optimizer_parameters.append(
                    {
                        "params": group_decay,
                        "lr": layer_lr,
                        "weight_decay": weight_decay,
                    }
                )
            if group_no_decay:
                optimizer_parameters.append(
                    {"params": group_no_decay, "lr": layer_lr, "weight_decay": 0.0}
                )

        # 2. Embeddings (Lowest LR)
        embeddings_lr = encoder_lr * (decay_rate**num_layers)
        embed_params = [p for n, p in self.backbone.embeddings.named_parameters()]
        # Also include relative embeddings if present and not in layers
        if hasattr(self.backbone, "rel_embeddings"):
            embed_params.extend(
                [
                    p
                    for n, p in self.backbone.named_parameters()
                    if "rel_embeddings" in n
                ]
            )

        optimizer_parameters.append(
            {"params": embed_params, "lr": embeddings_lr, "weight_decay": weight_decay}
        )

        # 3. Task Heads & Custom Layers (Highest LR = decoder_lr)
        # Includes pooler, weighted layer weights, regressor, classifier, etc.
        head_params = []
        for n, p in self.named_parameters():
            if "backbone" not in n:
                head_params.append((n, p))

        head_decay = [p for n, p in head_params if not any(nd in n for nd in no_decay)]
        head_no_decay = [p for n, p in head_params if any(nd in n for nd in no_decay)]

        if head_decay:
            optimizer_parameters.append(
                {"params": head_decay, "lr": decoder_lr, "weight_decay": weight_decay}
            )
        if head_no_decay:
            optimizer_parameters.append(
                {"params": head_no_decay, "lr": decoder_lr, "weight_decay": 0.0}
            )

        return optimizer_parameters
