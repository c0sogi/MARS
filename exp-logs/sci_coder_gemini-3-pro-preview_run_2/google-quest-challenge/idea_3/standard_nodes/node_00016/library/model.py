import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CausalAwareSiameseDeberta(nn.Module):
    """
    Causal-Aware Siamese DeBERTa Network.

    This architecture uses a shared DeBERTa-v3-base backbone to encode
    Questions and Answers independently. It enforces causal constraints by
    predicting question-related labels solely from the question embedding,
    while answer-related labels are predicted from a fused interaction vector.
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config()

        # Load Backbone Configuration and Model
        self.model_config = AutoConfig.from_pretrained(self.cfg.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(self.cfg.MODEL_NAME)

        # Hidden size of the backbone (e.g., 768 for base models)
        self.hidden_size = self.model_config.hidden_size

        # ------------------------------------------------------------------
        # Head 1: Question Head
        # Predicts 21 targets related to the question's intrinsic properties.
        # Input: Question Embedding u (Size: hidden_size)
        # ------------------------------------------------------------------
        self.question_head = nn.Linear(self.hidden_size, self.cfg.NUM_QUESTION_TARGETS)

        # ------------------------------------------------------------------
        # Head 2: Answer Interaction Head
        # Predicts 9 targets related to the answer's quality and relevance.
        # Input: Interaction Vector [u, v, |u-v|, u*v] (Size: 4 * hidden_size)
        # ------------------------------------------------------------------
        self.answer_head = nn.Linear(self.hidden_size * 4, self.cfg.NUM_ANSWER_TARGETS)

        # Initialize weights for the new classification heads
        self._init_weights(self.question_head)
        self._init_weights(self.answer_head)

    def _init_weights(self, module):
        """
        Initialize weights for linear layers using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Applies Mean Pooling to the token embeddings, ignoring padded tokens.

        Args:
            last_hidden_state: Tensor of shape (Batch, Seq_Len, Hidden_Size)
            attention_mask: Tensor of shape (Batch, Seq_Len)

        Returns:
            Tensor of shape (Batch, Hidden_Size)
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids_q,
        attention_mask_q,
        input_ids_a,
        attention_mask_a,
        token_type_ids_q=None,
        token_type_ids_a=None,
        **kwargs
    ):
        """
        Forward pass for the Siamese Network.

        Args:
            input_ids_q, attention_mask_q: Inputs for the Question stream.
            input_ids_a, attention_mask_a: Inputs for the Answer stream.
            token_type_ids_q, token_type_ids_a: Optional token type IDs (for DeBERTa).
            **kwargs: Catch-all for other arguments (e.g., labels) passed by the dataloader.

        Returns:
            logits: Tensor of shape (Batch, 30) containing unnormalized scores.
        """

        # ==========================================
        # 1. Question Stream Processing
        # ==========================================
        outputs_q = self.backbone(
            input_ids=input_ids_q,
            attention_mask=attention_mask_q,
            token_type_ids=token_type_ids_q,
        )
        # Obtain Question Embedding u via Mean Pooling
        u = self.mean_pooling(outputs_q.last_hidden_state, attention_mask_q)

        # ==========================================
        # 2. Answer Stream Processing
        # ==========================================
        outputs_a = self.backbone(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            token_type_ids=token_type_ids_a,
        )
        # Obtain Answer Embedding v via Mean Pooling
        v = self.mean_pooling(outputs_a.last_hidden_state, attention_mask_a)

        # ==========================================
        # 3. Prediction Heads
        # ==========================================

        # A. Question Head
        # Strictly causal: predictions depend ONLY on the question embedding u
        logits_q = self.question_head(u)

        # B. Answer Interaction Head
        # Relational: predictions depend on the interaction between u and v
        # Construct interaction vector: [u, v, |u-v|, u*v]
        diff_sim = torch.abs(u - v)
        prod_sim = u * v
        interaction_vec = torch.cat([u, v, diff_sim, prod_sim], dim=1)

        logits_a = self.answer_head(interaction_vec)

        # ==========================================
        # 4. Output Assembly
        # ==========================================
        # Concatenate logits to match the order of target columns in sample_submission.csv
        # (Question targets first, followed by Answer targets)
        logits = torch.cat([logits_q, logits_a], dim=1)

        return logits
