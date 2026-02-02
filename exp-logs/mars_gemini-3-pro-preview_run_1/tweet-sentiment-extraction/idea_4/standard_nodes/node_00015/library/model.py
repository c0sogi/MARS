import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling: Learns weights to aggregate the last N hidden layers.
    """

    def __init__(self, num_hidden_layers=4, hidden_size=768):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.weights = nn.Parameter(torch.ones(num_hidden_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of (Batch, Seq_Len, Hidden_Size)
        # We take the last 'num_hidden_layers'
        # Stack shape: (Num_Layers, Batch, Seq_Len, Hidden_Size)
        selected_layers = torch.stack(all_hidden_states[-self.num_hidden_layers :])

        # Normalize weights
        w = torch.softmax(self.weights, dim=0)

        # Reshape for broadcasting: (Num_Layers, 1, 1, 1)
        w = w.view(-1, 1, 1, 1)

        # Weighted sum: (Batch, Seq_Len, Hidden_Size)
        weighted_embedding = (w * selected_layers).sum(dim=0)
        return weighted_embedding


class TweetModel(nn.Module):
    def __init__(self, config):
        super(TweetModel, self).__init__()
        self.config = config

        # Load Backbone
        model_config = AutoConfig.from_pretrained(
            config.MODEL_NAME, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            config.MODEL_NAME, config=model_config
        )

        # Pooling
        self.pooling = WeightedLayerPooling(
            num_hidden_layers=config.N_POOLING_LAYERS, hidden_size=config.HIDDEN_SIZE
        )

        # Primary Head: 1D Convolution
        # We use kernel_size=1 to act as a token-wise projection,
        # but it's defined as Conv1d as per requirements.
        self.conv_head = nn.Conv1d(
            in_channels=config.HIDDEN_SIZE, out_channels=2, kernel_size=1
        )

        # Auxiliary Head: Dense Mask (Linear)
        if config.USE_AUX_HEAD:
            self.aux_head = nn.Linear(config.HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout(0.1)

        # Initialize Head Weights
        self._init_weights(self.conv_head)
        if config.USE_AUX_HEAD:
            self._init_weights(self.aux_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.HIDDEN_SIZE**-0.5)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        feature = self.pooling(all_hidden_states)
        return feature

    def forward(self, input_ids, attention_mask):
        # Extract features
        feature = self.feature(input_ids, attention_mask)  # (Batch, Seq, Hidden)
        feature = self.dropout(feature)

        # Primary Head (Conv1d)
        # Conv1d expects (Batch, Channels, Length)
        feature_transposed = feature.permute(0, 2, 1)
        logits = self.conv_head(feature_transposed)  # (Batch, 2, Length)

        # Split into start and end logits
        start_logits = logits[:, 0, :]  # (Batch, Length)
        end_logits = logits[:, 1, :]  # (Batch, Length)

        # Auxiliary Head
        mask_logits = None
        if self.config.USE_AUX_HEAD:
            mask_logits = self.aux_head(feature).squeeze(-1)  # (Batch, Length)

        return start_logits, end_logits, mask_logits
