import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Applies learned attention pooling to a sequence of hidden states,
    respecting a mask.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Seq, Hidden)
            mask: (Batch, Seq) - 1 for valid tokens, 0 for masked.
        Returns:
            pooled: (Batch, Hidden)
        """
        # Calculate attention scores
        w = self.attention(x)  # (B, S, 1)

        # Apply mask: set scores of masked tokens to -inf
        mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)
        w = w.masked_fill(mask_expanded == 0, -1e9)

        # Softmax to get weights
        weights = torch.softmax(w, dim=1)  # (B, S, 1)

        # Weighted sum
        # Note: If a sequence is fully masked, weights will be uniform (softmax of -inf)
        # but the mask multiplication in sum isn't explicit here.
        # However, since we use the weights derived from masked logits,
        # the weights for masked tokens are effectively 0.
        pooled = torch.sum(x * weights, dim=1)  # (B, H)

        return pooled


class SiameseDeberta(nn.Module):
    """
    Siamese Network with DeBERTa-v3 backbone and Decoupled Context-Response Pooling.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load Configuration and Backbone
        model_config = AutoConfig.from_pretrained(config.model_name)
        model_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": config.hidden_dropout_prob,
                "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
                "num_labels": 3,
            }
        )

        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=model_config
        )
        self.hidden_size = model_config.hidden_size

        # Pooling Mechanism
        self.pooler = AttentionPooling(self.hidden_size)

        # Classification Head
        # Features: Ra(H) + Rb(H) + Diff(H) + Prod(H) + Context(H) + Scalars(3)
        input_dim = 5 * self.hidden_size + 3

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, 3),
        )

        # Weight Initialization for custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_branch(self, input_ids, attention_mask, token_type_ids):
        """
        Process a single branch to extract Context and Response vectors.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract last 4 layers and average them
        # hidden_states is a tuple of (embeddings, layer1, ... layerN)
        # We take the last 4
        last_4_layers = outputs.hidden_states[-4:]
        stacked_layers = torch.stack(last_4_layers, dim=0)  # (4, B, S, H)
        avg_hidden = torch.mean(stacked_layers, dim=0)  # (B, S, H)

        # Decouple Context (Prompt) and Content (Response) using token_type_ids
        # Convention: Type 0 = Prompt, Type 1 = Response
        # We must also respect the padding mask (attention_mask == 1)

        prompt_mask = (token_type_ids == 0) & (attention_mask == 1)
        response_mask = (token_type_ids == 1) & (attention_mask == 1)

        # Apply Attention Pooling
        p_vector = self.pooler(avg_hidden, prompt_mask)
        r_vector = self.pooler(avg_hidden, response_mask)

        return p_vector, r_vector

    def forward(self, batch):
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(self.config.device)
        mask_a = batch["attention_mask_a"].to(self.config.device)
        type_a = batch["token_type_ids_a"].to(self.config.device)

        input_ids_b = batch["input_ids_b"].to(self.config.device)
        mask_b = batch["attention_mask_b"].to(self.config.device)
        type_b = batch["token_type_ids_b"].to(self.config.device)

        features = batch["features"].to(self.config.device)  # (B, 3)

        # Process Branch A
        p_a, r_a = self.forward_branch(input_ids_a, mask_a, type_a)

        # Process Branch B
        p_b, r_b = self.forward_branch(input_ids_b, mask_b, type_b)

        # Combine Context Vectors
        # Since the prompt is identical, we average the extracted context vectors
        # to get a more robust representation.
        p_context = (p_a + p_b) / 2.0

        # Interaction Features
        diff = torch.abs(r_a - r_b)
        prod = r_a * r_b

        # Concatenate all features
        # [Ra, Rb, |Ra-Rb|, Ra*Rb, Context, Scalars]
        combined = torch.cat([r_a, r_b, diff, prod, p_context, features], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits
