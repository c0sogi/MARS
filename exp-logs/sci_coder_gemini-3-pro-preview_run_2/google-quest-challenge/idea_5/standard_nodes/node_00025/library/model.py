import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class HybridDeberta(nn.Module):
    """
    Hybrid Multi-View DeBERTa architecture for Question-Answer subjective labeling.

    This model employs a shared DeBERTa-v3-base backbone to process data in two views:
    1. Intrinsic View: Processes Question Title + Body to predict 21 question-related labels.
       Ensures causal independence from the answer.
    2. Relational View: Processes Question + Answer to predict 9 answer-related labels.
       Utilizes cross-attention and explicit interaction features.
    """

    def __init__(self):
        super().__init__()
        # Load configuration and pre-trained backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name)

        hidden_size = self.config.hidden_size

        # ---------------------------------------------------------
        # Head 1: Intrinsic Question Processor (Causal-Aware)
        # ---------------------------------------------------------
        # Predicts 21 targets based solely on Question embedding
        self.head_intrinsic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(Config.hidden_dropout_prob),
            nn.Linear(hidden_size, len(Config.question_targets)),
        )

        # ---------------------------------------------------------
        # Head 2: Relational Cross-Encoder (Contextualized Interaction)
        # ---------------------------------------------------------
        # Predicts 9 targets based on fused interaction vector
        # Input dimension: 4 * hidden_size (u, v, |u-v|, u*v)
        self.head_relational = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(Config.hidden_dropout_prob),
            nn.Linear(hidden_size, len(Config.answer_targets)),
        )

        # Initialize weights for the custom heads
        self._init_weights(self.head_intrinsic)
        self._init_weights(self.head_relational)

    def _init_weights(self, module):
        """Initialize weights for the MLP heads."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        view1_input_ids,
        view1_attention_mask,
        view2_input_ids,
        view2_attention_mask,
        view2_q_mask,
        view2_a_mask,
        view2_token_type_ids=None,
        labels=None,
    ):
        """
        Forward pass handling both Intrinsic and Relational views.

        Args:
            view1_input_ids (Tensor): Input IDs for Question only.
            view1_attention_mask (Tensor): Attention mask for Question only.
            view2_input_ids (Tensor): Input IDs for Question + Answer pair.
            view2_attention_mask (Tensor): Attention mask for pair.
            view2_q_mask (Tensor): Binary mask identifying Question tokens in View 2.
            view2_a_mask (Tensor): Binary mask identifying Answer tokens in View 2.
            view2_token_type_ids (Tensor, optional): Token type IDs for View 2.
            labels (Tensor, optional): Targets (unused in forward, handled by loss function).

        Returns:
            Tensor: Logits of shape (Batch_Size, 30).
        """

        # =========================================================
        # View 1: Intrinsic Pass (Question Only)
        # =========================================================
        v1_outputs = self.backbone(
            input_ids=view1_input_ids, attention_mask=view1_attention_mask
        )
        v1_last_hidden = v1_outputs.last_hidden_state  # (B, Seq_Len, Hidden)

        # Mean Pooling for Intrinsic Representation (u_intrinsic)
        # Expand mask: (B, Seq_Len) -> (B, Seq_Len, Hidden)
        v1_mask_expanded = (
            view1_attention_mask.unsqueeze(-1).expand(v1_last_hidden.size()).float()
        )
        v1_sum = torch.sum(v1_last_hidden * v1_mask_expanded, 1)
        v1_counts = torch.clamp(v1_mask_expanded.sum(1), min=1e-9)
        u_intrinsic = v1_sum / v1_counts  # (B, Hidden)

        # Predict Question Targets (First 21 columns)
        q_logits = self.head_intrinsic(u_intrinsic)

        # =========================================================
        # View 2: Relational Pass (Question + Answer)
        # =========================================================
        v2_outputs = self.backbone(
            input_ids=view2_input_ids,
            attention_mask=view2_attention_mask,
            token_type_ids=view2_token_type_ids,
        )
        v2_last_hidden = v2_outputs.last_hidden_state  # (B, Seq_Len, Hidden)

        # Contextualized Question Pooling (u_ctx)
        # Use view2_q_mask to isolate Question tokens within the pair
        q_mask_expanded = (
            view2_q_mask.unsqueeze(-1).expand(v2_last_hidden.size()).float()
        )
        q_sum = torch.sum(v2_last_hidden * q_mask_expanded, 1)
        q_counts = torch.clamp(q_mask_expanded.sum(1), min=1e-9)
        u_ctx = q_sum / q_counts  # (B, Hidden)

        # Contextualized Answer Pooling (v_ctx)
        # Use view2_a_mask to isolate Answer tokens within the pair
        a_mask_expanded = (
            view2_a_mask.unsqueeze(-1).expand(v2_last_hidden.size()).float()
        )
        a_sum = torch.sum(v2_last_hidden * a_mask_expanded, 1)
        a_counts = torch.clamp(a_mask_expanded.sum(1), min=1e-9)
        v_ctx = a_sum / a_counts  # (B, Hidden)

        # Explicit Interaction Features
        diff_feat = torch.abs(u_ctx - v_ctx)
        prod_feat = u_ctx * v_ctx

        # Concatenate: [u_ctx, v_ctx, |u-v|, u*v] -> (B, 4*Hidden)
        interaction_vec = torch.cat([u_ctx, v_ctx, diff_feat, prod_feat], dim=1)

        # Predict Answer Targets (Last 9 columns)
        a_logits = self.head_relational(interaction_vec)

        # =========================================================
        # Output Assembly
        # =========================================================
        # Concatenate logits: (B, 21) + (B, 9) -> (B, 30)
        logits = torch.cat([q_logits, a_logits], dim=1)

        return logits
