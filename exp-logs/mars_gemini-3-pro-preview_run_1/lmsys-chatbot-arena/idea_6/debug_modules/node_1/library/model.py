import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted sum of the last N hidden layers.
    Returns a sequence representation (Batch, SeqLen, Hidden).
    """

    def __init__(self, num_hidden_layers=12, layer_start=8, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.num_layers_to_mix = (
            num_hidden_layers - layer_start + 1
        )  # +1 includes the final embedding

        # Learnable weights for the layers
        self.layer_weights = nn.Parameter(torch.tensor([1.0] * self.num_layers_to_mix))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (initial_embeds, layer_1, ..., layer_12)
        # We are interested in the last N layers.
        # Note: AutoModel output.hidden_states includes the initial embeddings at index 0.
        # So layer 1 is at index 1.

        # Select the layers we want to mix
        # If layer_start is 9 (1-based from config usually means last 4 layers of 12 -> 9,10,11,12)
        # In 0-indexing tuple: indices 9, 10, 11, 12.

        # Config.USE_LAST_N_LAYERS = 4.
        # Total layers in base = 12. Output tuple size = 13.
        # We want indices [-4:].

        selected_layers = all_hidden_states[-Config.USE_LAST_N_LAYERS :]

        # Stack: (Batch, Seq, Hidden, NumLayers)
        stacked_layers = torch.stack(selected_layers, dim=-1)

        # Normalize weights
        weights = F.softmax(self.layer_weights, dim=0)

        # Weighted sum: (Batch, Seq, Hidden)
        # sum( H_i * w_i )
        weighted_sum = (stacked_layers * weights.view(1, 1, 1, -1)).sum(dim=-1)

        return weighted_sum


class CrossAttentionLayer(nn.Module):
    """
    A single layer of Cross-Attention followed by a Feed-Forward Network.
    Performs Attention(Q=x, K=context, V=context).
    """

    def __init__(self, hidden_size, num_heads, dropout_prob):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, context, key_padding_mask=None):
        """
        Args:
            x: Query sequence (Batch, SeqLen_Q, Hidden)
            context: Key/Value sequence (Batch, SeqLen_KV, Hidden)
            key_padding_mask: Boolean mask for context (Batch, SeqLen_KV), True where padded.
        """
        # Cross Attention
        # attn_output: (Batch, SeqLen_Q, Hidden)
        attn_output, _ = self.multihead_attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Residual + Norm
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class SiameseDebertaCrossAttn(nn.Module):
    """
    Siamese DeBERTa-v3-Base with Weighted Layer Mixing and Late-Stage Cross-Attention.
    """

    def __init__(self):
        super(SiameseDebertaCrossAttn, self).__init__()

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # 2. Weighted Layer Pooling
        # DeBERTa Base has 12 layers.
        # Calculate layer_start based on how many last layers we want to use
        layer_start = self.config.num_hidden_layers - Config.USE_LAST_N_LAYERS + 1
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers, layer_start=layer_start
        )

        # 3. Cross Attention Blocks
        # We share the weights for A->B and B->A interaction to maintain symmetry
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    hidden_size=Config.HIDDEN_SIZE,
                    num_heads=Config.CROSS_ATTN_HEADS,
                    dropout_prob=Config.DROPOUT_PROB,
                )
                for _ in range(Config.CROSS_ATTN_LAYERS)
            ]
        )

        # 4. Classifier Head
        # Input: Pooled(A) + Pooled(B) + Scalar Features
        self.scalar_dim = 6  # Defined in data_processing.py
        self.fusion_dim = (Config.HIDDEN_SIZE * 2) + self.scalar_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, Config.HIDDEN_SIZE),
            nn.LayerNorm(Config.HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_PROB),
            nn.Linear(Config.HIDDEN_SIZE, Config.NUM_CLASSES),
        )

        # Initialize weights for new layers
        self._init_weights(self.cross_attn_layers)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.ModuleList):
            for submodule in module:
                self._init_weights(submodule)

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalar_features,
    ):
        # --- 1. Siamese Encoding ---
        # Branch A
        out_a = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        # Branch B
        out_b = self.backbone(input_ids=input_ids_b, attention_mask=attention_mask_b)

        # --- 2. Weighted Layer Mixing ---
        # Get sequence representations (Batch, Seq, Hidden)
        seq_a = self.pooler(out_a.hidden_states)
        seq_b = self.pooler(out_b.hidden_states)

        # --- 3. Cross Attention ---
        # In PyTorch MultiheadAttention, key_padding_mask expects True for padded positions.
        # Our attention_mask is 1 for keep, 0 for pad.
        # So we invert it: mask == 0 -> True (pad)
        pad_mask_a = attention_mask_a == 0
        pad_mask_b = attention_mask_b == 0

        # Iterative interaction
        # We update seq_a by attending to seq_b, and seq_b by attending to seq_a
        curr_a = seq_a
        curr_b = seq_b

        for layer in self.cross_attn_layers:
            # Calculate updates
            # Note: We use the *current* state of the other branch as context
            next_a = layer(curr_a, curr_b, key_padding_mask=pad_mask_b)
            next_b = layer(curr_b, curr_a, key_padding_mask=pad_mask_a)

            curr_a = next_a
            curr_b = next_b

        # --- 4. Mean Pooling ---
        # Pool the *attended* sequences
        # Mask out padding tokens for correct mean calculation

        def mean_pooling(hidden_states, attention_mask):
            # hidden_states: (B, L, D)
            # attention_mask: (B, L)
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

        pool_a = mean_pooling(curr_a, attention_mask_a)
        pool_b = mean_pooling(curr_b, attention_mask_b)

        # --- 5. Fusion & Classification ---
        # Concatenate: [Pool_A, Pool_B, Scalars]
        combined = torch.cat([pool_a, pool_b, scalar_features], dim=1)

        logits = self.classifier(combined)

        return logits
