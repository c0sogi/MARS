import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomModel(nn.Module):
    """
    Custom model class wrapping DeBERTa-v3-large with a Multi-Sample Dropout head.
    """

    def __init__(self):
        super().__init__()
        self.model_name = Config.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)

        # Ensure we output hidden states if we want to do specific pooling,
        # though for CLS/Mean pooling on last_hidden_state it's default.
        self.config.output_hidden_states = True

        # Initialize Backbone
        self.backbone = AutoModel.from_pretrained(self.model_name, config=self.config)

        # Gradient Checkpointing (Optional, helps with memory on large batches)
        # self.backbone.gradient_checkpointing_enable()

        # Multi-Sample Dropout Head
        self.num_msd = Config.num_msd_rounds
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.fc_dropout) for _ in range(self.num_msd)]
        )

        # Regression Head (Linear projection to scalar score)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.

        Returns:
            torch.Tensor: Predicted similarity scores.
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Pooling Strategy: Mean Pooling
        # We use the last hidden state and mask out padding tokens
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Multi-Sample Dropout
        # Pass the pooled embedding through multiple dropout masks and average the predictions
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.fc(dropout(pooled_output))
            else:
                output += self.fc(dropout(pooled_output))

        output /= self.num_msd

        return output

    def get_optimizer_params(self, base_lr, weight_decay, layer_decay):
        """
        Groups parameters for Layer-wise Learning Rate Decay (LLRD).

        Args:
            base_lr (float): The learning rate for the head (top layer).
            weight_decay (float): Weight decay coefficient.
            layer_decay (float): Multiplicative decay factor for lower layers.

        Returns:
            list: List of parameter groups for the optimizer.
        """
        # DeBERTa-v3 structure: embeddings -> encoder.layer.0 ... encoder.layer.23

        # 1. Identify layers
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []

        # Initialize layer names
        # Embeddings are considered layer 0 effectively for decay purposes relative to the first encoder layer
        # But typically we treat embeddings as the "bottom-most" layer.

        # We'll assign IDs to layers.
        # Head = Max ID
        # Encoder 23 = Max ID - 1
        # ...
        # Encoder 0
        # Embeddings

        # Get all named parameters
        named_parameters = list(self.named_parameters())

        # Define layer mapping
        # DeBERTa Large has 24 layers
        num_layers = self.config.num_hidden_layers

        # Function to determine layer id
        def get_layer_id(name):
            if "embeddings" in name:
                return 0
            elif "encoder.layer" in name:
                # Extract number from "encoder.layer.X."
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "layer":
                        return int(parts[i + 1]) + 1
            else:
                # Head or other top-level params
                return num_layers + 1

        # Group parameters
        for name, param in named_parameters:
            if not param.requires_grad:
                continue

            layer_id = get_layer_id(name)

            # Calculate LR for this layer
            # Head (layer_id = num_layers + 1) gets base_lr
            # Layer N gets base_lr * (layer_decay ^ (num_layers + 1 - layer_id))
            scale = layer_decay ** (num_layers + 1 - layer_id)
            group_lr = base_lr * scale

            # Check weight decay
            if any(nd in name for nd in no_decay):
                group_decay = 0.0
            else:
                group_decay = weight_decay

            optimizer_grouped_parameters.append(
                {"params": [param], "lr": group_lr, "weight_decay": group_decay}
            )

        return optimizer_grouped_parameters
