import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted average of the last N hidden layers.
    """

    def __init__(self, num_hidden_layers, layer_start=9):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights to be equal
        self.layer_weights = nn.Parameter(
            torch.tensor([1.0] * (num_hidden_layers - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (batch, seq_len, hidden)
        # Stack the desired layers: (num_layers_to_pool, batch, seq_len, hidden)
        hidden_states = torch.stack(all_hidden_states[self.layer_start :], dim=0)

        # Compute softmax weights
        weights = F.softmax(self.layer_weights, dim=0)

        # Weighted sum: (batch, seq_len, hidden)
        weighted_pooling_embeddings = torch.sum(
            weights.view(-1, 1, 1, 1) * hidden_states, dim=0
        )
        return weighted_pooling_embeddings


class CoAttentionLayer(nn.Module):
    """
    Performs decomposable attention (Co-Attention) between two sequences
    and computes interaction features.
    """

    def __init__(self, hidden_size):
        super(CoAttentionLayer, self).__init__()
        self.hidden_size = hidden_size

        # Project the 4x hidden size interaction vector back to hidden size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, h1, h2, mask1, mask2):
        """
        Enhance h1 with context from h2.
        Args:
            h1: (Batch, Len1, Hidden) - The sequence to enhance
            h2: (Batch, Len2, Hidden) - The context sequence
            mask1: (Batch, Len1) - Mask for h1
            mask2: (Batch, Len2) - Mask for h2
        """
        # 1. Compute Affinity Matrix: (Batch, Len1, Len2)
        # Scale by sqrt(hidden_size)
        scores = torch.matmul(h1, h2.transpose(-1, -2)) / (self.hidden_size**0.5)

        # 2. Apply Mask for h2 (columns)
        # mask2 is 1 for valid, 0 for pad. We want -inf for pad.
        mask2_expanded = mask2.unsqueeze(1).float()  # (Batch, 1, Len2)
        scores = scores.masked_fill(mask2_expanded == 0, -1e9)

        # 3. Attention Weights
        attn_weights = F.softmax(scores, dim=-1)  # (Batch, Len1, Len2)

        # 4. Compute Context (Weighted sum of h2)
        context = torch.matmul(attn_weights, h2)  # (Batch, Len1, Hidden)

        # 5. Interaction Features
        # [h1, context, |h1 - context|, h1 * context]
        interaction = torch.cat(
            [h1, context, torch.abs(h1 - context), h1 * context], dim=-1
        )  # (Batch, Len1, 4*Hidden)

        # 6. Project back
        output = self.projection(interaction)  # (Batch, Len1, Hidden)

        return output


class MultiSampleDropout(nn.Module):
    """
    Applies multiple dropout masks to the input and averages the predictions
    to improve generalization.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_samples=5, dropout_rate=0.1
    ):
        super(MultiSampleDropout, self).__init__()
        self.num_samples = num_samples
        self.dropout = nn.Dropout(dropout_rate)

        # Feature extraction part
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()

        # Prediction part
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        # Shared feature extraction
        h = self.linear1(x)
        h = self.ln(h)
        h = self.gelu(h)

        # Multi-sample dropout and prediction
        logits_list = []
        for _ in range(self.num_samples):
            d = self.dropout(h)
            logits_list.append(self.linear2(d))

        # Average logits
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)


class SiameseCoAttentionNetwork(nn.Module):
    """
    Main model class implementing Siamese DeBERTa with Granular Co-Attention Fusion.
    """

    def __init__(self):
        super(SiameseCoAttentionNetwork, self).__init__()

        self.model_name = Config.model_name
        self.hidden_size = Config.hidden_size
        self.num_targets = Config.num_targets

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(
            self.model_name, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(self.model_name, config=self.config)

        # 2. Weighted Layer Pooling
        # DeBERTa-v3-base has 12 layers. hidden_states tuple has 13 elements (emb + 12 layers).
        # We use the last 4 layers.
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers + 1,
            layer_start=self.config.num_hidden_layers + 1 - 4,
        )

        # 3. Co-Attention Layer
        # Shared layer for all interactions to learn a common interaction space
        self.co_attn = CoAttentionLayer(self.hidden_size)

        # 4. Categorical Embeddings
        # category: 5 unique values + 1 for safety
        self.cat_emb = nn.Embedding(6, 16)
        # host: 63 unique values + 1 for safety
        self.host_emb = nn.Embedding(64, 32)

        # 5. Prediction Head
        # Input Dimension Calculation:
        # We have 4 interaction streams: (A-T), (A-B), (T-A), (B-A)
        # Each stream is pooled using Mean and Max pooling -> 2 vectors per stream
        # Total pooled vectors: 4 streams * 2 pools = 8 vectors
        # Dimension: 8 * hidden_size + 16 (cat) + 32 (host)
        head_input_dim = (8 * self.hidden_size) + 16 + 32

        self.head = MultiSampleDropout(
            input_dim=head_input_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.num_targets,
            num_samples=5,
            dropout_rate=Config.dropout,
        )

        self._init_emb_weights()

    def _init_emb_weights(self):
        for m in [self.cat_emb, self.host_emb]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

    def _pool(self, x, mask):
        """
        Applies Mean and Max pooling to sequence x based on mask.
        Returns concatenated vector of size 2*hidden.
        """
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)

        # Mean Pooling
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Max Pooling
        # Fill masked positions with very small number
        x_masked = x.masked_fill(mask_expanded == 0, -1e9)
        max_pooled = torch.max(x_masked, dim=1)[0]

        return torch.cat([mean_pooled, max_pooled], dim=1)

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        q_token_type_ids,
        a_input_ids,
        a_attention_mask,
        cats,
    ):

        # --- 1. Backbone Encoding ---
        # Question Stream
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = self.pooler(q_out.hidden_states)  # (B, Lq, D)

        # Answer Stream
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = self.pooler(a_out.hidden_states)  # (B, La, D)

        # --- 2. Granular Masking ---
        # Split Question hidden states into Title and Body using token_type_ids
        # 0 = Title, 1 = Body
        title_mask = q_attention_mask * (q_token_type_ids == 0)
        body_mask = q_attention_mask * (q_token_type_ids == 1)

        # --- 3. Co-Attention Fusion ---

        # Interaction 1: Enhance Answer with Title Context (A <- T)
        a_enhanced_t = self.co_attn(a_hidden, q_hidden, a_attention_mask, title_mask)

        # Interaction 2: Enhance Answer with Body Context (A <- B)
        a_enhanced_b = self.co_attn(a_hidden, q_hidden, a_attention_mask, body_mask)

        # Interaction 3: Enhance Title with Answer Context (T <- A)
        t_enhanced_a = self.co_attn(q_hidden, a_hidden, title_mask, a_attention_mask)

        # Interaction 4: Enhance Body with Answer Context (B <- A)
        b_enhanced_a = self.co_attn(q_hidden, a_hidden, body_mask, a_attention_mask)

        # --- 4. Pooling ---
        # Pool the enhanced sequences using their respective valid masks

        # Pool Answer representations (valid tokens only)
        pool_a_t = self._pool(a_enhanced_t, a_attention_mask)
        pool_a_b = self._pool(a_enhanced_b, a_attention_mask)

        # Pool Question representations (Title/Body parts only)
        pool_t_a = self._pool(t_enhanced_a, title_mask)
        pool_b_a = self._pool(b_enhanced_a, body_mask)

        # --- 5. Metadata ---
        cat_vec = self.cat_emb(cats[:, 0])
        host_vec = self.host_emb(cats[:, 1])

        # --- 6. Concatenation & Prediction ---
        features = torch.cat(
            [pool_a_t, pool_a_b, pool_t_a, pool_b_a, cat_vec, host_vec], dim=1
        )

        logits = self.head(features)
        probs = torch.sigmoid(logits)

        return probs
