import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class QuestModel(nn.Module):
    """
    DistilRoBERTa Dual-Encoder model for StackExchange Question-Answer classification.
    """

    def __init__(self):
        super(QuestModel, self).__init__()

        # Configuration
        self.model_name = Config.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)

        # Backbone
        self.backbone = AutoModel.from_pretrained(self.model_name, config=self.config)

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
        sequence_output = outputs.last_hidden_state

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
