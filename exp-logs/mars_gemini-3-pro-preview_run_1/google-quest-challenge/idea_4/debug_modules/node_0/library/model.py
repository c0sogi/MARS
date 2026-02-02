import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted average of the last `num_hidden_layers` from the backbone.
    """

    def __init__(self, num_hidden_layers=4):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.weights = nn.Parameter(torch.ones(num_hidden_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (embeddings, layer_1, ..., layer_N)
        # We extract the last 'num_hidden_layers'
        selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack to shape: (batch, seq_len, hidden_size, num_layers)
        stacked = torch.stack(selected_layers, dim=-1)

        # Normalize weights using Softmax
        w = F.softmax(self.weights, dim=0)

        # Compute weighted sum along the last dimension
        # (batch, seq_len, hidden, num_layers) * (1, 1, 1, num_layers) -> sum over last dim
        weighted_output = (stacked * w.view(1, 1, 1, -1)).sum(dim=-1)

        return weighted_output


class QuestModel(nn.Module):
    """
    DeBERTa-v3 Dual-Encoder model for StackExchange Question-Answer classification.
    """

    def __init__(self):
        super(QuestModel, self).__init__()

        # Configuration
        self.model_name = Config.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.config.output_hidden_states = True

        # Backbone
        self.backbone = AutoModel.from_pretrained(self.model_name, config=self.config)

        # Weighted Pooling
        self.pooler = WeightedLayerPooling(num_hidden_layers=Config.num_pooling_layers)

        # Fusion Dimension Calculation
        # We concatenate: u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|
        # Total = 6 vectors of size hidden_size
        self.fusion_dim = Config.hidden_size * 6

        # Normalization
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # Monolithic Prediction Head (MLP)
        self.fc = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(self.fusion_dim // 2, Config.num_classes),
        )

        # Initialize custom layers
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for sub_module in module:
                self._init_weights(sub_module)

    def feature_extraction(self, input_ids, attention_mask):
        """
        Extracts features for a single branch (Question or Answer).
        Returns both Average Pooled and Max Pooled vectors.
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Apply Weighted Layer Pooling -> (batch, seq_len, hidden)
        sequence_output = self.pooler(hidden_states)

        # Create masks
        # attention_mask shape: (batch, seq_len)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        )

        # Masked Global Average Pooling
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        avg_pool = sum_embeddings / sum_mask

        # Masked Global Max Pooling
        # Replace padding tokens with a very small number so they aren't picked by max()
        sequence_output_masked = sequence_output.clone()
        sequence_output_masked[input_mask_expanded == 0] = -1e9
        max_pool = torch.max(sequence_output_masked, 1)[0]

        return avg_pool, max_pool

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass for the dual-encoder.
        """
        # 1. Process Question Branch
        u_avg, u_max = self.feature_extraction(q_input_ids, q_attention_mask)

        # 2. Process Answer Branch
        v_avg, v_max = self.feature_extraction(a_input_ids, a_attention_mask)

        # 3. Interaction-Aware Fusion
        # Interactions computed only on Average Pooled representations
        prod = u_avg * v_avg
        diff = torch.abs(u_avg - v_avg)

        # 4. Concatenation
        # [u_avg, u_max, v_avg, v_max, prod, diff]
        features = torch.cat([u_avg, u_max, v_avg, v_max, prod, diff], dim=1)

        # 5. Normalization
        features = self.layer_norm(features)

        # 6. Prediction Head
        logits = self.fc(features)

        return logits
