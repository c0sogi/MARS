import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SiameseDebertaWithScalars(nn.Module):
    """
    A Siamese Network architecture using a DeBERTa backbone with densely injected
    structural scalar features.

    Architecture:
    1. Shared DeBERTa Encoder for Branch A and Branch B.
    2. Extraction of [CLS] tokens (u, v).
    3. Computation of interaction terms: |u-v| and u*v.
    4. Concatenation of [u, v, |u-v|, u*v, scalar_features].
    5. MLP Classification Head to predict logits for 3 classes.
    """

    def __init__(self):
        super(SiameseDebertaWithScalars, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)

        # Initialize Pre-trained Backbone
        # We use the shared backbone for both branches of the Siamese network
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Calculate Input Dimension for the Classification Head
        # The head receives a concatenation of:
        # 1. Embedding u (hidden_size)
        # 2. Embedding v (hidden_size)
        # 3. Diff |u - v| (hidden_size)
        # 4. Prod u * v (hidden_size)
        # 5. Scalar Features (NUM_SCALAR_FEATURES)
        self.hidden_size = self.config.hidden_size

        head_input_dim = (4 * self.hidden_size) + Config.NUM_SCALAR_FEATURES

        # Define Classification Head (MLP)
        # Projects the high-dimensional feature vector down to the number of classes
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.HIDDEN_DROPOUT_PROB),
            nn.Linear(self.hidden_size, 3),  # Output: [logit_A, logit_B, logit_Tie]
        )

        # Initialize weights for the new head layers
        self._init_head_weights()

    def _init_head_weights(self):
        """
        Applies initialization to the classification head layers.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalar_features,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids_a (torch.Tensor): Input IDs for response A (batch, seq_len).
            attention_mask_a (torch.Tensor): Attention mask for response A (batch, seq_len).
            input_ids_b (torch.Tensor): Input IDs for response B (batch, seq_len).
            attention_mask_b (torch.Tensor): Attention mask for response B (batch, seq_len).
            scalar_features (torch.Tensor): Pre-computed scalar features (batch, num_scalars).

        Returns:
            torch.Tensor: Logits for the 3 classes (batch, 3).
        """
        # --- Branch A ---
        outputs_a = self.backbone(
            input_ids=input_ids_a, attention_mask=attention_mask_a
        )
        last_hidden_state_a = outputs_a.last_hidden_state
        # Mean Pooling for Branch A
        input_mask_expanded_a = (
            attention_mask_a.unsqueeze(-1).expand(last_hidden_state_a.size()).float()
        )
        u = torch.sum(last_hidden_state_a * input_mask_expanded_a, 1) / torch.clamp(
            input_mask_expanded_a.sum(1), min=1e-9
        )

        # --- Branch B ---
        outputs_b = self.backbone(
            input_ids=input_ids_b, attention_mask=attention_mask_b
        )
        last_hidden_state_b = outputs_b.last_hidden_state
        # Mean Pooling for Branch B
        input_mask_expanded_b = (
            attention_mask_b.unsqueeze(-1).expand(last_hidden_state_b.size()).float()
        )
        v = torch.sum(last_hidden_state_b * input_mask_expanded_b, 1) / torch.clamp(
            input_mask_expanded_b.sum(1), min=1e-9
        )

        # --- Interaction Terms ---
        # Absolute difference captures distance in semantic space
        diff_uv = torch.abs(u - v)
        # Element-wise product captures similarity/alignment
        prod_uv = u * v

        # --- Feature Fusion ---
        # Concatenate all semantic and structural signals
        # scalar_features shape: (batch_size, Config.NUM_SCALAR_FEATURES)
        combined_features = torch.cat([u, v, diff_uv, prod_uv, scalar_features], dim=1)

        # --- Classification ---
        logits = self.classifier(combined_features)

        return logits
