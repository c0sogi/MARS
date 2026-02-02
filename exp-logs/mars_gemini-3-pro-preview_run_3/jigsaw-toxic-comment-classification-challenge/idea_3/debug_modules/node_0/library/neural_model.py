import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Performs mean pooling on the last hidden state of the transformer,
    accounting for the attention mask to ignore padding tokens.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        # Expand mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings (masking padding)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Calculate sum of mask (number of valid tokens)
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class DebertaClassifier(nn.Module):
    """
    DeBERTa-v3-Large based classifier with Mean Pooling and LLRD support.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_labels=Config.NUM_LABELS):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooling = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

        # Initialize weights for the classification head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Pass through backbone
        # DeBERTa V3 handles token_type_ids if provided
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Pooling
        pool_out = self.pooling(last_hidden_state, attention_mask)

        # Classification
        logits = self.fc(pool_out)

        return logits

    def get_optimizer_params(self, base_lr, weight_decay, llrd_decay):
        """
        Generates optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD).

        Args:
            base_lr (float): The learning rate for the top layer (head).
            weight_decay (float): Weight decay coefficient.
            llrd_decay (float): Decay rate for lower layers (e.g., 0.9).

        Returns:
            list: List of parameter groups for the optimizer.
        """
        # Parameters to exclude from weight decay
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Initialize groups
        # Keys: 'head', 'embeddings', or layer index (int)
        groups = {}

        named_parameters = list(self.named_parameters())
        num_layers = self.config.num_hidden_layers

        for name, param in named_parameters:
            if not param.requires_grad:
                continue

            # Determine group based on parameter name
            if "backbone" not in name:
                # Parameters in self.fc (head)
                group_name = "head"
            elif "embeddings" in name:
                # Embeddings (word_embeddings, position_embeddings, etc.)
                group_name = "embeddings"
            elif "encoder.layer" in name:
                # Transformer layers
                # Format usually: backbone.encoder.layer.X.output...
                parts = name.split(".")
                try:
                    # Find the index following 'layer'
                    layer_idx_pos = parts.index("layer") + 1
                    layer_idx = int(parts[layer_idx_pos])
                    group_name = layer_idx
                except (ValueError, IndexError):
                    # Fallback for unexpected naming in encoder
                    group_name = "embeddings"
            else:
                # Other backbone parameters (e.g. final LayerNorm, rel_embeddings if outside layer)
                group_name = "embeddings"

            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append((name, param))

        optimizer_grouped_parameters = []

        # 1. Head (Highest LR)
        head_lr = base_lr
        head_params = groups.get("head", [])

        decay_params = [
            p for n, p in head_params if not any(nd in n for nd in no_decay)
        ]
        nodecay_params = [p for n, p in head_params if any(nd in n for nd in no_decay)]

        if decay_params:
            optimizer_grouped_parameters.append(
                {"params": decay_params, "lr": head_lr, "weight_decay": weight_decay}
            )
        if nodecay_params:
            optimizer_grouped_parameters.append(
                {"params": nodecay_params, "lr": head_lr, "weight_decay": 0.0}
            )

        # 2. Transformer Layers (Decaying LR from top to bottom)
        # Layer N-1 (top) to Layer 0 (bottom)
        for i in range(num_layers - 1, -1, -1):
            # Calculate decayed LR
            # Layer depth from top: (num_layers - 1) is depth 0 relative to encoder top
            # We want: Top Layer = base_lr * decay
            # Next Layer = base_lr * decay^2
            depth = num_layers - i
            layer_lr = base_lr * (llrd_decay**depth)

            layer_params = groups.get(i, [])

            decay_params = [
                p for n, p in layer_params if not any(nd in n for nd in no_decay)
            ]
            nodecay_params = [
                p for n, p in layer_params if any(nd in n for nd in no_decay)
            ]

            if decay_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": decay_params,
                        "lr": layer_lr,
                        "weight_decay": weight_decay,
                    }
                )
            if nodecay_params:
                optimizer_grouped_parameters.append(
                    {"params": nodecay_params, "lr": layer_lr, "weight_decay": 0.0}
                )

        # 3. Embeddings (Lowest LR)
        embed_lr = base_lr * (llrd_decay ** (num_layers + 1))
        embed_params = groups.get("embeddings", [])

        decay_params = [
            p for n, p in embed_params if not any(nd in n for nd in no_decay)
        ]
        nodecay_params = [p for n, p in embed_params if any(nd in n for nd in no_decay)]

        if decay_params:
            optimizer_grouped_parameters.append(
                {"params": decay_params, "lr": embed_lr, "weight_decay": weight_decay}
            )
        if nodecay_params:
            optimizer_grouped_parameters.append(
                {"params": nodecay_params, "lr": embed_lr, "weight_decay": 0.0}
            )

        return optimizer_grouped_parameters
