import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted average of hidden states based on a learned attention score.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [Batch, Seq, Hidden]
        # attention_mask: [Batch, Seq] (1 for valid tokens, 0 for pad/ignored)

        # Calculate attention scores
        w = self.attention(hidden_states)  # [Batch, Seq, 1]
        w = w.squeeze(-1)  # [Batch, Seq]

        # Mask padding and ignored tokens
        # We use a large negative number so softmax results in 0 for these positions
        w = w.float().masked_fill(attention_mask == 0, -1e9)

        # Normalize weights
        weights = torch.softmax(w, dim=-1)  # [Batch, Seq]
        weights = weights.unsqueeze(-1)  # [Batch, Seq, 1]

        # Weighted sum
        pooled = torch.sum(hidden_states * weights, dim=1)  # [Batch, Hidden]
        return pooled


class SiameseDeberta(nn.Module):
    """
    Siamese Network with Disentangled Hierarchical Pooling.
    Uses DeBERTa-v3-base backbone with separate pooling strategies for Prompt and Response.
    """

    def __init__(self):
        super(SiameseDeberta, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True
        self.config.hidden_dropout_prob = Config.DROPOUT
        self.config.attention_probs_dropout_prob = Config.DROPOUT

        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Load tokenizer to get special token IDs (not in config for DeBERTa V3)
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.sep_token_id = tokenizer.sep_token_id

        # Enable Gradient Checkpointing for memory efficiency
        if Config.USE_GRADIENT_CHECKPOINTING:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.hidden_size = self.config.hidden_size

        # --- Pooling Layers ---
        # Response Stream: Extracts features from the last 4 layers -> 4 independent poolers
        self.response_poolers = nn.ModuleList(
            [AttentionPooling(self.hidden_size) for _ in range(4)]
        )

        # Context Stream: Extracts features from the last layer only -> 1 pooler
        self.context_pooler = AttentionPooling(self.hidden_size)

        # --- Classification Head ---
        # Input Dimension Calculation:
        # 1. Response Vector R: 4 layers * 768 = 3072
        # 2. Interaction Features: R_a, R_b, |R_a - R_b|, R_a * R_b -> 4 * 3072 = 12288
        # 3. Context Vector P: 768
        # 4. Scalars: 3
        # Total Input: 12288 + 768 + 3 = 13059

        input_dim = (4 * 4 * self.hidden_size) + self.hidden_size + 3

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(1024, Config.NUM_LABELS),
        )

    def _get_disentangled_masks(self, input_ids, attention_mask):
        """
        Creates separate masks for Prompt and Response based on the first [SEP] token.
        Assumes input structure: [CLS] Prompt [SEP] Response [SEP]
        """
        sep_token_id = self.sep_token_id

        # Find the index of the first [SEP] token for each sequence in the batch
        # (input_ids == sep_token_id) returns a boolean tensor
        # .long().argmax(dim=1) returns the index of the first '1' (True)
        sep_indices = (input_ids == sep_token_id).long().argmax(dim=1).unsqueeze(1)

        # Create a range tensor to compare against indices
        seq_len = input_ids.size(1)
        indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Prompt Mask: Include tokens up to and including the first [SEP]
        # Also must respect the original padding mask
        prompt_mask = (indices <= sep_indices) & (attention_mask.bool())

        # Response Mask: Include tokens after the first [SEP]
        response_mask = (indices > sep_indices) & (attention_mask.bool())

        return prompt_mask.long(), response_mask.long()

    def _process_branch(self, input_ids, attention_mask):
        """
        Processes a single branch (A or B) to extract Response and Context vectors.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Tuple of tensors

        # Generate masks to separate content (Response) from context (Prompt)
        prompt_mask, response_mask = self._get_disentangled_masks(
            input_ids, attention_mask
        )

        # --- Response Stream ---
        # Extract last 4 layers, pool independently using response_mask, and concatenate
        response_features = []
        # Indices for last 4 layers: -1, -2, -3, -4
        for i, layer_idx in enumerate([-1, -2, -3, -4]):
            layer_hidden = hidden_states[layer_idx]
            pooler = self.response_poolers[i]
            pooled = pooler(layer_hidden, response_mask)
            response_features.append(pooled)

        r_vector = torch.cat(response_features, dim=1)  # [Batch, 3072]

        # --- Context Stream ---
        # Extract last layer only, pool using prompt_mask
        last_hidden = hidden_states[-1]
        p_vector = self.context_pooler(last_hidden, prompt_mask)  # [Batch, 768]

        return r_vector, p_vector

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalars,
        labels=None,
    ):
        # Process Branch A
        r_a, p_a = self._process_branch(input_ids_a, attention_mask_a)

        # Process Branch B
        r_b, p_b = self._process_branch(input_ids_b, attention_mask_b)

        # Combine Context Vectors (Average)
        # Since the prompt is identical, averaging reduces noise
        p_context = (p_a + p_b) / 2.0

        # Interaction Features
        diff = torch.abs(r_a - r_b)
        prod = r_a * r_b

        # Concatenate all features
        # [R_A, R_B, |R_A-R_B|, R_A*R_B, P, Scalars]
        features = torch.cat([r_a, r_b, diff, prod, p_context, scalars], dim=1)

        # Classification
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"logits": logits, "loss": loss}
