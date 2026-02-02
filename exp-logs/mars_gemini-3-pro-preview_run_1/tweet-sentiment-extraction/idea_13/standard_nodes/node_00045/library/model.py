import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling.
    Dynamically learns weights to aggregate hidden states from the last N layers.
    """

    def __init__(self, num_hidden_layers=4, layer_start=20, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * num_hidden_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # Select the last N layers
        # all_hidden_states is a tuple of tensors (batch, seq_len, hidden)
        # We take the last num_hidden_layers
        all_layer_embedding = torch.stack(all_hidden_states[-self.num_hidden_layers :])

        # (num_layers, batch, seq_len, hidden)

        # Calculate soft weights
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )
        weight_factor = torch.nn.functional.softmax(weight_factor, dim=0)

        # Weighted sum
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0)
        return weighted_average


class TweetModel(nn.Module):
    """
    DeBERTa-v3-Large with Shared CNN Head.
    """

    def __init__(self):
        super(TweetModel, self).__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_PATH)
        self.config.output_hidden_states = True

        self.backbone = AutoModel.from_pretrained(Config.MODEL_PATH, config=self.config)

        # Weighted Layer Pooling for the last 4 layers
        self.pooler = WeightedLayerPooling(num_hidden_layers=4)

        # Shared CNN Head
        # Input to Conv1d: (Batch, Channels/Hidden, Seq_Len)
        self.cnn = nn.Conv1d(
            in_channels=Config.HIDDEN_SIZE,
            out_channels=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            padding=(Config.CNN_KERNEL_SIZE - 1) // 2,
        )

        self.dropout = nn.Dropout(Config.DROP_RATE)

        # Final projection to Start/End logits
        self.fc = nn.Linear(Config.CNN_FILTERS, 2)

        self._init_weights(self.cnn)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # 1. Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 2. Weighted Pooling
        # Shape: (Batch, Seq_Len, Hidden)
        feature = self.pooler(all_hidden_states)

        # 3. Shared CNN Layer
        # Permute for Conv1d: (Batch, Hidden, Seq_Len)
        feature = feature.permute(0, 2, 1)
        feature = self.cnn(feature)
        # Permute back: (Batch, Seq_Len, Hidden)
        feature = feature.permute(0, 2, 1)

        # 4. Activation (Optional, but often implicitly linear in simple heads,
        # usually ReLU is between convs, but here we just have one conv as context extractor)
        # We'll apply GELU as it matches DeBERTa's internal activations
        feature = torch.nn.functional.gelu(feature)

        # 5. Dropout & Projection
        feature = self.dropout(feature)
        logits = self.fc(feature)

        # 6. Split Logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
