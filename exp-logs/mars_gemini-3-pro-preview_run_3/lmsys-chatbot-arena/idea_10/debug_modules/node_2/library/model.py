import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted average of token embeddings based on a learned attention score.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, mask):
        """
        Args:
            x: Hidden states (Batch, Seq, Hidden)
            mask: Binary mask (Batch, Seq), 1 for valid tokens, 0 for ignore.
        Returns:
            Pooled vector (Batch, Hidden)
        """
        # Compute raw attention scores
        w = self.attention(x)  # (B, Seq, 1)
        w = w.squeeze(-1)  # (B, Seq)

        # Mask padding/irrelevant tokens by setting score to -inf
        # mask is 1 for keep, 0 for ignore
        # Cast to float32 to avoid overflow with -1e9 in mixed precision
        w = w.float().masked_fill(mask == 0, -1e9)

        # Softmax to get probability distribution
        weights = torch.softmax(w, dim=1)  # (B, Seq)

        # Weighted sum of hidden states
        # (B, Seq, 1) * (B, Seq, H) -> Sum over Seq
        out = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return out


class SiameseDeberta(nn.Module):
    """
    Siamese Network with DeBERTa-v3-Large backbone and Decoupled Contextual Pooling.
    """

    def __init__(self):
        super(SiameseDeberta, self).__init__()

        # Load Configuration
        self.model_name = Config.model_name
        self.num_classes = Config.num_classes
        self.dropout_rate = Config.dropout

        # Load Backbone
        # We use output_hidden_states=True to access the last 4 layers
        config = AutoConfig.from_pretrained(self.model_name)
        config.output_hidden_states = True
        config.attention_probs_dropout_prob = 0.0  # Reduce internal noise
        config.hidden_dropout_prob = 0.0

        self.backbone = AutoModel.from_pretrained(self.model_name, config=config)

        # Enable Gradient Checkpointing for memory efficiency with Large model
        if Config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.hidden_size = config.hidden_size

        # Pooling Layers
        # We pool the last 4 layers separately for both Prompt and Response
        # 4 layers * hidden_size
        self.poolers_prompt = nn.ModuleList(
            [AttentionPooling(self.hidden_size) for _ in range(4)]
        )
        self.poolers_response = nn.ModuleList(
            [AttentionPooling(self.hidden_size) for _ in range(4)]
        )

        # Feature Dimension Calculation
        # Each stream (Prompt or Response) is a concatenation of 4 pooled layers
        self.stream_dim = self.hidden_size * 4

        # Final Feature Vector Components:
        # 1. Response A (stream_dim)
        # 2. Response B (stream_dim)
        # 3. |Response A - Response B| (stream_dim)
        # 4. Response A * Response B (stream_dim)
        # 5. Context P (stream_dim)
        # 6. Context P * Response A (stream_dim)
        # 7. Context P * Response B (stream_dim)
        # Total Vector Dim = 7 * stream_dim
        # Plus 3 scalars

        self.feature_dim = (7 * self.stream_dim) + 3

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, self.num_classes),
        )

        # Initialize custom layers
        self._init_weights(self.poolers_prompt)
        self._init_weights(self.poolers_response)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ModuleList):
            for m in module:
                self._init_weights(m)
        elif isinstance(module, nn.Sequential):
            for m in module:
                self._init_weights(m)

    def get_stream_embeddings(self, input_ids, attention_mask, token_type_ids):
        """
        Processes one branch to obtain Prompt and Response embeddings.
        """
        # Forward pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get hidden states from last 4 layers
        # outputs.hidden_states is a tuple of (layer_0, ..., layer_N)
        # We take the last 4
        hidden_states = outputs.hidden_states[-4:]

        # Create Masks
        # token_type_ids: 0 for Prompt, 1 for Response (usually)
        # We enforce valid tokens only using attention_mask
        mask_prompt = (token_type_ids == 0) & (attention_mask == 1)
        mask_response = (token_type_ids == 1) & (attention_mask == 1)

        # Pool Prompt Stream
        # Apply specific pooler to each layer's output
        prompt_vecs = []
        for i, layer_out in enumerate(hidden_states):
            p = self.poolers_prompt[i](layer_out, mask_prompt)
            prompt_vecs.append(p)

        # Pool Response Stream
        response_vecs = []
        for i, layer_out in enumerate(hidden_states):
            r = self.poolers_response[i](layer_out, mask_response)
            response_vecs.append(r)

        # Concatenate layers -> (B, 4*H)
        P = torch.cat(prompt_vecs, dim=1)
        R = torch.cat(response_vecs, dim=1)

        return P, R

    def forward(self, batch):
        # Unpack batch
        input_ids_a = batch["input_ids_a"]
        attention_mask_a = batch["attention_mask_a"]
        token_type_ids_a = batch["token_type_ids_a"]

        input_ids_b = batch["input_ids_b"]
        attention_mask_b = batch["attention_mask_b"]
        token_type_ids_b = batch["token_type_ids_b"]

        scalars = batch["scalars"]  # (B, 3)

        # Branch A
        P_a, R_a = self.get_stream_embeddings(
            input_ids_a, attention_mask_a, token_type_ids_a
        )

        # Branch B
        P_b, R_b = self.get_stream_embeddings(
            input_ids_b, attention_mask_b, token_type_ids_b
        )

        # Aggregate Context (Prompt) Vector
        # Average P_a and P_b to get a stable representation of the prompt
        P = (P_a + P_b) / 2.0

        # Feature Engineering
        # 1. Interaction Features
        diff_sim = torch.abs(R_a - R_b)
        prod_sim = R_a * R_b

        # 2. Context Modulation
        ctx_a = P * R_a
        ctx_b = P * R_b

        # Concatenate all features
        # [R_a, R_b, |Ra-Rb|, Ra*Rb, P, P*Ra, P*Rb]
        combined_features = torch.cat(
            [R_a, R_b, diff_sim, prod_sim, P, ctx_a, ctx_b], dim=1
        )

        # Append Scalars
        final_input = torch.cat([combined_features, scalars], dim=1)

        # Classification
        logits = self.classifier(final_input)

        return logits
