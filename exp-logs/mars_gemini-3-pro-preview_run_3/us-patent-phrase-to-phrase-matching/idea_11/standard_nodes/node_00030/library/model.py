import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import CFG


class DynamicLayerMixing(nn.Module):
    """
    Learns scalar weights to mix hidden states from all layers of the transformer.
    Formula: H_mix = sum(softmax(w) * H_i)
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Initialize weights to be equal (0.0 -> softmax -> 1/N)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tuple of (Batch, Seq, Dim) tensors from the backbone.
        Returns:
            Tensor of shape (Batch, Seq, Dim) representing the weighted average.
        """
        # Stack hidden states: (Batch, Seq, Dim, Layers)
        # We stack along the last dimension for easier broadcasting with weights
        all_layers = torch.stack(hidden_states, dim=-1)

        # Compute normalized weights via softmax
        weights = torch.softmax(self.layer_weights, dim=0)  # Shape: (Layers)

        # Weighted sum
        # Reshape weights to (1, 1, 1, Layers) to broadcast
        # (B, S, D, L) * (1, 1, 1, L) -> Sum over L -> (B, S, D)
        mixed_layer = (all_layers * weights.view(1, 1, 1, -1)).sum(dim=-1)

        return mixed_layer


class AttentionPooling(nn.Module):
    """
    Aggregates token embeddings using a learned attention mechanism.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (Batch, Seq, Dim)
            attention_mask: (Batch, Seq)
        Returns:
            Tensor of shape (Batch, Dim)
        """
        # Compute attention scores
        w = self.attention(last_hidden_state)  # (Batch, Seq, 1)

        # Mask padding tokens (set to very small value so softmax -> 0)
        mask = attention_mask.unsqueeze(-1)  # (Batch, Seq, 1)
        w = w.masked_fill(mask == 0, -1e4)

        # Normalize weights across sequence length
        w = torch.softmax(w, dim=1)  # (Batch, Seq, 1)

        # Weighted sum of token embeddings
        weighted_embeddings = torch.sum(last_hidden_state * w, dim=1)  # (Batch, Dim)

        return weighted_embeddings


class PhraseModel(nn.Module):
    """
    Main model architecture for Phrase Matching.

    Structure:
    1. Backbone (DeBERTa-v3)
    2. Dynamic Layer Mixing (Weighted sum of all layers)
    3. Attention Pooling
    4. Multi-Sample Dropout
    5. Dual Heads (Regression + Classification)
    """

    def __init__(self, model_name=None, pretrained=True):
        super().__init__()
        if model_name is None:
            model_name = CFG.model_name

        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.config.use_cache = False

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing for memory efficiency if configured
        if CFG.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Determine number of layers (Embeddings + Encoder Layers)
        # DeBERTa output_hidden_states usually contains N+1 tensors
        self.num_layers = self.config.num_hidden_layers + 1

        # --- Custom Components ---
        self.layer_mixing = DynamicLayerMixing(self.num_layers)
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout: 5 parallel dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(CFG.fc_dropout) for _ in range(5)])

        # Heads
        self.fc_reg = nn.Linear(self.config.hidden_size, CFG.target_size)
        self.fc_cls = nn.Linear(self.config.hidden_size, CFG.num_classes)

        # Initialize weights for custom modules
        self._init_weights(self.pooler)
        self._init_weights(self.fc_reg)
        self._init_weights(self.fc_cls)

    def _init_weights(self, module):
        """
        Initialize weights for custom layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for sub_module in module:
                self._init_weights(sub_module)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        labels_cls=None,
    ):
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Retrieve all hidden states (Tuple of tensors)
        hidden_states = outputs.hidden_states

        # 1. Dynamic Layer Mixing
        # Aggregates info from all layers into one sequence
        mixed_features = self.layer_mixing(hidden_states)  # (Batch, Seq, Dim)

        # 2. Attention Pooling
        # Aggregates sequence into a single vector
        pooled_features = self.pooler(mixed_features, attention_mask)  # (Batch, Dim)

        # 3. Multi-Sample Dropout & Heads
        reg_logits_list = []
        cls_logits_list = []

        for dropout in self.dropouts:
            dropped = dropout(pooled_features)

            # Regression Output
            reg_logits_list.append(self.fc_reg(dropped))

            # Classification Output
            cls_logits_list.append(self.fc_cls(dropped))

        # Average the predictions from the multiple dropout samples
        # Stack: (5, B, 1) -> Mean: (B, 1)
        reg_logits = torch.mean(torch.stack(reg_logits_list, dim=0), dim=0)
        # Stack: (5, B, 5) -> Mean: (B, 5)
        cls_logits = torch.mean(torch.stack(cls_logits_list, dim=0), dim=0)

        return {
            "logits": reg_logits.squeeze(-1),  # Flatten to (Batch)
            "logits_cls": cls_logits,  # (Batch, Num_Classes)
        }
