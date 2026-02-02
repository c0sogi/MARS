import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Performs mean pooling on the token embeddings, accounting for the attention mask.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # expand mask to match hidden state dimensions: [batch, seq_len, hidden_dim]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings ignoring padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Count non-padding tokens (clamp to avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask


class SiameseModel(nn.Module):
    """
    Siamese Network using DeBERTa-v3-base backbone.
    Processes (Prompt, Response A) and (Prompt, Response B) separately,
    then combines embeddings with interaction terms and meta-features.
    """

    def __init__(self):
        super(SiameseModel, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": 0.0,  # We handle dropout in the head
                "attention_probs_dropout_prob": 0.0,
            }
        )
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable Gradient Checkpointing for memory efficiency if needed
        # self.backbone.gradient_checkpointing_enable()

        self.pooling = MeanPooling()

        # Feature Dimensions
        # u, v, |u-v|, u*v -> 4 * hidden_size
        # meta_features -> 3 (norm_len_prompt, norm_len_a, norm_len_b)
        self.hidden_size = self.config.hidden_size
        self.meta_dim = 3
        self.combined_dim = (4 * self.hidden_size) + self.meta_dim

        # Multi-Sample Dropout settings
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.DROP_RATE) for _ in range(Config.NUM_DROPOUT_SAMPLES)]
        )

        # Classification Head
        self.fc = nn.Linear(self.combined_dim, 3)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def extract_embedding(self, input_ids, attention_mask):
        """
        Forward pass for one branch of the Siamese network.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        embedding = self.pooling(last_hidden_state, attention_mask)
        return embedding

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        meta_features,
    ):
        # 1. Extract embeddings for both branches
        u = self.extract_embedding(input_ids_a, attention_mask_a)
        v = self.extract_embedding(input_ids_b, attention_mask_b)

        # 2. Interaction Features
        diff_abs = torch.abs(u - v)
        prod = u * v

        # 3. Concatenate all features
        # [Batch, 4*Hidden + Meta]
        features = torch.cat([u, v, diff_abs, prod, meta_features], dim=1)

        # 4. Multi-Sample Dropout and Classification
        # Average the logits from multiple dropout masks
        logits_list = []
        for dropout in self.dropouts:
            dropped_features = dropout(features)
            logits_list.append(self.fc(dropped_features))

        # Stack and mean
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits


def get_llrd_optimizer_params(model):
    """
    Configures Layer-wise Learning Rate Decay (LLRD) for the optimizer.

    Groups parameters into:
    1. Embeddings (lowest LR)
    2. Encoder Layers (increasing LR from bottom to top)
    3. Head/Pooler (highest LR = Config.LEARNING_RATE)

    Returns:
        List of dictionaries containing params and specific learning rates.
    """

    # Base learning rate and decay factor
    init_lr = Config.LEARNING_RATE
    decay_rate = Config.LLRD_DECAY
    weight_decay = Config.WEIGHT_DECAY

    # DeBERTa-v3-base usually has 12 layers.
    # Structure: embeddings -> encoder.layer.0 ... encoder.layer.11 -> head

    # Identify the number of layers from config
    num_hidden_layers = model.config.num_hidden_layers

    # Initialize groups
    # Group 0: Embeddings
    # Group 1..N: Encoder layers
    # Group N+1: Head/Pooler

    # Calculate LR for each layer: LR_i = init_lr * (decay_rate ^ (num_layers + 1 - i))
    # Head is at index (num_hidden_layers + 1), so exponent is 0 -> init_lr

    optimizer_grouped_parameters = []

    # 1. Embeddings
    # Typically treated as layer 0 in terms of depth relative to encoder stack
    embed_lr = init_lr * (decay_rate ** (num_hidden_layers + 1))
    embed_params = []

    # 2. Encoder Layers
    # List of lists to hold params for each layer
    layer_params = [[] for _ in range(num_hidden_layers)]

    # 3. Head / Top parameters
    head_params = []

    # Iterate through named parameters and assign to groups
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "embeddings" in name:
            embed_params.append(param)
        elif "encoder.layer" in name:
            # Extract layer index. Example: backbone.encoder.layer.5.output...
            # Split by dot and find the integer after 'layer'
            parts = name.split(".")
            found_layer = False
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    layer_params[layer_idx].append(param)
                    found_layer = True
                    break
            if not found_layer:
                # Fallback if naming convention differs (unlikely for HF DeBERTa)
                head_params.append(param)
        else:
            # Classification head, pooling, rel_embeddings, etc.
            head_params.append(param)

    # Create parameter groups

    # Group: Embeddings
    optimizer_grouped_parameters.append(
        {"params": embed_params, "lr": embed_lr, "weight_decay": weight_decay}
    )

    # Group: Encoder Layers (0 to 11)
    for layer_idx in range(num_hidden_layers):
        # Decay exponent: Head is 0 decay. Layer 11 is 1 decay. Layer 0 is 12 decay.
        # exponent = num_hidden_layers - layer_idx
        lr = init_lr * (decay_rate ** (num_hidden_layers - layer_idx))
        optimizer_grouped_parameters.append(
            {"params": layer_params[layer_idx], "lr": lr, "weight_decay": weight_decay}
        )

    # Group: Head
    optimizer_grouped_parameters.append(
        {"params": head_params, "lr": init_lr, "weight_decay": weight_decay}
    )

    return optimizer_grouped_parameters
