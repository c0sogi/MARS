import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Pools the last N hidden layers using learnable scalar weights.
    Extracts the [CLS] token (index 0) from each layer.
    """

    def __init__(self, num_hidden_layers=4):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.weight = nn.Parameter(torch.tensor([1.0] * num_hidden_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors from the backbone
        # We take the last 'num_hidden_layers'
        # Shape of each layer tensor: (batch_size, seq_len, hidden_size)

        selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack CLS tokens: (batch_size, num_layers, hidden_size)
        # We assume the CLS token is at index 0 (standard for BERT/RoBERTa/DeBERTa)
        cls_embeddings = torch.stack(
            [layer[:, 0, :] for layer in selected_layers], dim=1
        )

        # Compute softmax weights: (num_layers,)
        weights = F.softmax(self.weight, dim=0)

        # Weighted sum: (batch_size, hidden_size)
        # Broadcast weights: (1, num_layers, 1)
        weighted_embeddings = (cls_embeddings * weights.view(1, -1, 1)).sum(dim=1)

        return weighted_embeddings


class StylometricFusionModel(nn.Module):
    """
    Multi-modal architecture fusing Transformer text embeddings with dense stylometric features.
    """

    def __init__(
        self,
        backbone_name,
        num_classes=Config.NUM_CLASSES,
        num_style_features=13,
        dropout_p=0.1,
    ):
        super(StylometricFusionModel, self).__init__()

        # 1. Text Backbone
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(backbone_name, config=self.config)

        self.hidden_size = self.config.hidden_size

        # 2. Text Pooling
        self.pooling = WeightedLayerPooling(num_hidden_layers=4)

        # 3. Style Branch
        # Project dense features (dim=13) to latent space (dim=64)
        self.style_dim = 64
        self.style_mlp = nn.Sequential(
            nn.Linear(num_style_features, self.style_dim),
            nn.BatchNorm1d(self.style_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        # 4. Fusion Head
        self.fusion_dim = self.hidden_size + self.style_dim

        # Multi-Sample Dropout
        # Applies dropout k times and averages the results to improve generalization
        self.num_dropouts = 5
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_p) for _ in range(self.num_dropouts)]
        )
        self.fc = nn.Linear(self.fusion_dim, num_classes)

        # Initialize custom weights
        self._init_weights(self.style_mlp)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom heads.
        """
        if isinstance(module, nn.Linear):
            # Use the backbone's initializer range if available, else default
            std = getattr(self.config, "initializer_range", 0.02)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for submodule in module:
                self._init_weights(submodule)

    def forward(self, input_ids, attention_mask, style_features):
        """
        Forward pass.
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            style_features: (batch, num_style_features)
        Returns:
            logits: (batch, num_classes)
        """
        # --- Text Path ---
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.hidden_states is a tuple of hidden states
        text_embedding = self.pooling(outputs.hidden_states)  # (batch, hidden_size)

        # --- Style Path ---
        style_embedding = self.style_mlp(style_features)  # (batch, style_dim)

        # --- Fusion ---
        fused_embedding = torch.cat(
            (text_embedding, style_embedding), dim=1
        )  # (batch, hidden + style)

        # --- Multi-Sample Dropout Head ---
        # Apply multiple dropouts and average the predictions
        logits_list = []
        for dropout in self.dropouts:
            dropped = dropout(fused_embedding)
            logits_list.append(self.fc(dropped))

        # Stack and mean: (num_dropouts, batch, classes) -> (batch, classes)
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits
