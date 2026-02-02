import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class ResponseIsolatedPooling(nn.Module):
    """
    Applies attention-based pooling specifically to response tokens,
    ignoring prompt and padding tokens.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, response_mask):
        """
        Args:
            last_hidden_state: (Batch, SeqLen, HiddenSize)
            response_mask: (Batch, SeqLen) - 1 for response tokens, 0 for others.
        """
        # Calculate raw attention scores: (Batch, SeqLen)
        w = self.attention(last_hidden_state).squeeze(-1)

        # Masking: Set scores for non-response tokens (prompt/pad) to -inf
        # response_mask is 1.0 for valid response tokens, 0.0 otherwise.
        # We mask where response_mask == 0.
        mask_value = -1e9
        w = w.masked_fill(response_mask == 0, mask_value)

        # Apply Softmax to get normalized weights over the response tokens
        w = F.softmax(w, dim=1)

        # Weighted sum of hidden states
        # Expand weights for broadcasting: (Batch, SeqLen, 1)
        w = w.unsqueeze(-1)
        pooled = torch.sum(last_hidden_state * w, dim=1)  # (Batch, HiddenSize)

        return pooled


class SiameseDeberta(nn.Module):
    """
    Siamese Architecture using DeBERTa-v3-Large backbone with
    Response-Isolated Pooling and Hybrid Features.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable Gradient Checkpointing to save VRAM with Large models
        if Config.GRADIENT_CHECKPOINTING:
            self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooling = ResponseIsolatedPooling(self.config.hidden_size)

        # Calculate Input Dimension for the Classifier
        # We concatenate: u, v, |u-v|, u*v -> 4 vectors
        feature_dim = 4 * self.config.hidden_size

        # Add Scalar features dimension if enabled
        if Config.USE_SCALAR_FEATURES:
            feature_dim += 3  # log_len_prompt, log_len_resp_a, log_len_resp_b

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, Config.NUM_CLASSES),
        )

    def forward_branch(self, input_ids, attention_mask, response_mask):
        """
        Processes a single branch (A or B) through the shared backbone and pooling.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Pool only the response tokens
        pooled_output = self.pooling(last_hidden_state, response_mask)
        return pooled_output

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        response_mask_a,
        input_ids_b,
        attention_mask_b,
        response_mask_b,
        scalars=None,
    ):
        """
        Forward pass for the Siamese model.
        """
        # 1. Encode Branch A
        u = self.forward_branch(input_ids_a, attention_mask_a, response_mask_a)

        # 2. Encode Branch B
        v = self.forward_branch(input_ids_b, attention_mask_b, response_mask_b)

        # 3. Compute Interaction Features
        diff = torch.abs(u - v)
        prod = u * v

        # 4. Concatenate Features
        features = [u, v, diff, prod]

        if Config.USE_SCALAR_FEATURES:
            if scalars is None:
                raise ValueError("Scalars required but not provided.")
            features.append(scalars)

        combined_features = torch.cat(features, dim=1)

        # 5. Classification
        logits = self.classifier(combined_features)

        return logits
