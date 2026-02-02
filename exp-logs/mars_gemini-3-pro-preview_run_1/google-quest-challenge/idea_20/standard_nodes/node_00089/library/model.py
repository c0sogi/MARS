import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class SharedBottomMultiBranchModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load Configuration and Base Model
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        base_model = AutoModel.from_pretrained(Config.MODEL_NAME)

        # 1. Embeddings
        self.embeddings = base_model.embeddings

        # 2. Shared Bottom Encoder (Layers 0-9)
        # We extract the layers from the base model
        all_layers = base_model.encoder.layer
        self.shared_layers = nn.ModuleList(
            [all_layers[i] for i in range(Config.SHARED_LAYER_COUNT)]
        )

        # 3. Independent Top Encoders (Layers 10-11)
        # Initialize as copies of the original top layers
        split_layers_indices = range(
            Config.SHARED_LAYER_COUNT, config.num_hidden_layers
        )
        source_split_layers = [all_layers[i] for i in split_layers_indices]

        self.q_branch = nn.ModuleList(
            [copy.deepcopy(layer) for layer in source_split_layers]
        )
        self.a_branch = nn.ModuleList(
            [copy.deepcopy(layer) for layer in source_split_layers]
        )

        # 4. Interaction and Head Setup
        self.hidden_size = Config.HIDDEN_SIZE

        # Fused Vector F Components:
        # u_avg, u_max, v_avg, v_max (4 vectors)
        # u_avg * v_avg (1 vector)
        # |u_avg - v_avg| (1 vector)
        # Total = 6 vectors
        self.fusion_dim = self.hidden_size * 6

        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # Residual Interaction Head
        # Structure: Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
        self.head_proj = nn.Linear(self.fusion_dim, self.hidden_size)
        self.head_dropout = nn.Dropout(Config.DROPOUT)
        self.head_final = nn.Linear(
            self.fusion_dim + self.hidden_size, Config.NUM_TARGETS
        )

        # Initialize Head Weights
        self._init_weights(self.head_proj)
        self._init_weights(self.head_final)

    def _init_weights(self, module):
        """Initialize weights with Normal distribution as per strategy."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=Config.INIT_MEAN, std=Config.INIT_STD)
            if module.bias is not None:
                module.bias.data.zero_()

    def _get_extended_attention_mask(self, attention_mask, dtype):
        """
        Converts 2D attention mask [batch, seq] to 4D additive mask [batch, 1, 1, seq]
        compatible with RoBERTa/BERT attention scores.
        1.0 in mask indicates keep, 0.0 indicates mask.
        Result: 0.0 for keep, -10000.0 for mask.
        """
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask.to(dtype=dtype)

    def _process_layers(self, hidden_states, attention_mask, layers):
        """Passes hidden states through a list of transformer layers."""
        dtype = hidden_states.dtype
        extended_mask = self._get_extended_attention_mask(attention_mask, dtype)

        for layer in layers:
            # Transformer layer output is (hidden_states, attention_weights, ...)
            layer_outputs = layer(hidden_states, attention_mask=extended_mask)
            hidden_states = layer_outputs[0]

        return hidden_states

    def masked_mean_pooling(self, hidden_states, attention_mask):
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def masked_max_pooling(self, hidden_states, attention_mask):
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        # Set padded tokens to large negative value so they aren't picked as max
        hidden_states = hidden_states.clone()
        hidden_states[mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(hidden_states, 1)
        return max_embeddings

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a):
        # 1. Embeddings
        q_emb = self.embeddings(input_ids=input_ids_q)
        a_emb = self.embeddings(input_ids=input_ids_a)

        # 2. Shared Bottom (Layers 0-9)
        q_shared = self._process_layers(q_emb, attention_mask_q, self.shared_layers)
        a_shared = self._process_layers(a_emb, attention_mask_a, self.shared_layers)

        # 3. Independent Top (Layers 10-11)
        q_out = self._process_layers(q_shared, attention_mask_q, self.q_branch)
        a_out = self._process_layers(a_shared, attention_mask_a, self.a_branch)

        # 4. Pooling
        q_avg = self.masked_mean_pooling(q_out, attention_mask_q)
        q_max = self.masked_max_pooling(q_out, attention_mask_q)

        a_avg = self.masked_mean_pooling(a_out, attention_mask_a)
        a_max = self.masked_max_pooling(a_out, attention_mask_a)

        # 5. Interaction (Only on Average Pooled vectors)
        prod = q_avg * a_avg
        diff = torch.abs(q_avg - a_avg)

        # 6. Fusion
        F_vec = torch.cat([q_avg, q_max, a_avg, a_max, prod, diff], dim=1)
        F_vec = self.layer_norm(F_vec)

        # 7. Residual Interaction Head
        # Path A: Non-linear projection
        proj = self.head_proj(F_vec)
        proj = F.relu(proj)
        proj = self.head_dropout(proj)

        # Path B: Skip connection (Concatenation)
        concat = torch.cat([F_vec, proj], dim=1)

        # Final Logits
        logits = self.head_final(concat)

        return logits
