import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from copy import deepcopy
from library.config import Config


class ResidualProjectionHead(nn.Module):
    """
    Residual Projection Block:
    Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        # The skip connection concatenates the original input with the transformed latent
        self.linear2 = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, x):
        # x: [Batch, Input_Dim]
        latent = self.linear1(x)
        latent = self.relu(latent)
        latent = self.dropout(latent)

        # Residual Fusion: Concatenate original features with latent features
        cat = torch.cat([x, latent], dim=1)

        # Final projection to targets
        logits = self.linear2(cat)
        return logits


class SharedBottomSplitTopRoBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(Config.model_name)
        base_model = AutoModel.from_pretrained(Config.model_name)

        self.embeddings = base_model.embeddings

        # Shared Layers: First 10 layers (0-9)
        self.shared_layers = nn.ModuleList(base_model.encoder.layer[:10])

        # Split Layers: Top 2 layers (10-11)
        # We deepcopy them to create independent branches initialized with pretrained weights
        self.q_top = nn.ModuleList(deepcopy(base_model.encoder.layer[10:]))
        self.a_top = nn.ModuleList(deepcopy(base_model.encoder.layer[10:]))

        # Keep a reference to base_model for helper methods like get_extended_attention_mask
        self.base_model_ref = base_model

        self.hidden_size = config.hidden_size

        # Feature Construction
        # We concatenate:
        # 1. u_title (Mean)
        # 2. u_body (Mean)
        # 3. v_answer (Mean)
        # 4. I_intent (Product)
        # 5. I_intent (Difference)
        # 6. I_context (Product)
        # 7. I_context (Difference)
        # 8. u_title (Max)
        # 9. u_body (Max)
        # 10. v_answer (Max)
        # Total = 10 vectors
        input_dim = self.hidden_size * 10

        self.layer_norm = nn.LayerNorm(input_dim)

        # Residual Head
        # Inner dimension matches hidden size
        self.head = ResidualProjectionHead(
            input_dim, self.hidden_size, Config.num_labels
        )

    def _masked_pool(self, hidden, mask):
        """
        Performs Mean and Max pooling with explicit masking.
        """
        # mask: (B, L) -> (B, L, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Mean Pooling
        sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid div by zero
        avg_pool = sum_embeddings / sum_mask

        # Max Pooling
        # Set masked positions to a very small number
        hidden_masked = hidden.clone()
        hidden_masked[mask == 0] = -1e9
        max_pool = torch.max(hidden_masked, dim=1)[0]

        return avg_pool, max_pool

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        # 1. Embeddings
        q_emb = self.embeddings(q_input_ids)
        a_emb = self.embeddings(a_input_ids)

        # 2. Create Extended Attention Masks (for Transformer layers)
        q_mask = self.base_model_ref.get_extended_attention_mask(
            q_attention_mask, q_input_ids.shape
        )
        a_mask = self.base_model_ref.get_extended_attention_mask(
            a_attention_mask, a_input_ids.shape
        )

        # 3. Shared Bottom Pass (Layers 0-9)
        q_hidden = q_emb
        a_hidden = a_emb

        for layer in self.shared_layers:
            q_hidden = layer(q_hidden, attention_mask=q_mask)[0]
            a_hidden = layer(a_hidden, attention_mask=a_mask)[0]

        # 4. Split Top Pass (Layers 10-11)
        for layer in self.q_top:
            q_hidden = layer(q_hidden, attention_mask=q_mask)[0]

        for layer in self.a_top:
            a_hidden = layer(a_hidden, attention_mask=a_mask)[0]

        # 5. Partitioned Pooling (Question Branch)
        # We need to separate Title and Body based on SEP tokens (id=2)
        # Structure: <s> Title </s> </s> Body </s>
        sep_id = 2
        sep_mask = (q_input_ids == sep_id).long()
        # Cumulative sum helps identify segments:
        # Before 1st SEP: 0 (Title)
        # At/After 1st SEP, Before 2nd SEP: 1 (Gap)
        # At/After 2nd SEP: 2 (Body)
        segment_ids = torch.cumsum(sep_mask, dim=1)

        # Title Mask: segment_ids == 0, excluding <s> (id=0)
        title_mask = (segment_ids == 0) & (q_input_ids != 0) & (q_attention_mask == 1)

        # Body Mask: segment_ids == 2, excluding the SEP itself (id=2)
        body_mask = (segment_ids == 2) & (q_input_ids != 2) & (q_attention_mask == 1)

        u_title_avg, u_title_max = self._masked_pool(q_hidden, title_mask)
        u_body_avg, u_body_max = self._masked_pool(q_hidden, body_mask)

        # 6. Pooling (Answer Branch)
        # Standard pooling excluding special tokens (0 and 2)
        ans_mask = (a_input_ids != 0) & (a_input_ids != 2) & (a_attention_mask == 1)
        v_ans_avg, v_ans_max = self._masked_pool(a_hidden, ans_mask)

        # 7. Geometric Interactions
        # Intent Matching: Title vs Answer
        i_intent_prod = u_title_avg * v_ans_avg
        i_intent_diff = torch.abs(u_title_avg - v_ans_avg)

        # Context Matching: Body vs Answer
        i_context_prod = u_body_avg * v_ans_avg
        i_context_diff = torch.abs(u_body_avg - v_ans_avg)

        # 8. Feature Concatenation
        features = torch.cat(
            [
                u_title_avg,
                u_body_avg,
                v_ans_avg,
                i_intent_prod,
                i_intent_diff,
                i_context_prod,
                i_context_diff,
                u_title_max,
                u_body_max,
                v_ans_max,
            ],
            dim=1,
        )

        # 9. Normalization & Head
        features = self.layer_norm(features)
        logits = self.head(features)

        return logits
