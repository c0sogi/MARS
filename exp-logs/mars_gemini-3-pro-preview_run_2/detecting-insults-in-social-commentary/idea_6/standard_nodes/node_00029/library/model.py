import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class InsultModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.config = Config

        # Load Configuration
        self.model_config = AutoConfig.from_pretrained(self.config.model_name)
        self.model_config.output_hidden_states = True

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                self.config.model_name, config=self.model_config
            )
        else:
            self.backbone = AutoModel.from_config(self.model_config)

        # Enable Gradient Checkpointing to save memory
        # self.backbone.gradient_checkpointing_enable()

        # Classification Head
        self.fc = nn.Linear(self.model_config.hidden_size, self.config.num_classes)
        self._init_weights(self.fc)

        # Dropout layers
        if self.config.use_msd:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(p) for p in self.config.msd_rates]
            )
        else:
            self.std_dropout = nn.Dropout(self.config.fc_dropout)

    def _init_weights(self, module):
        """
        Custom weight initialization for the head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def pool(self, last_hidden_state, attention_mask):
        """
        Pools the token embeddings to a single sentence embedding.
        """
        if self.config.pooler_type == "mean":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

        elif self.config.pooler_type == "cls":
            return last_hidden_state[:, 0, :]

        elif self.config.pooler_type == "max":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            # Set padding tokens to large negative value
            last_hidden_state[input_mask_expanded == 0] = -1e9
            return torch.max(last_hidden_state, 1)[0]

        else:
            raise ValueError(f"Invalid pooler type: {self.config.pooler_type}")

    def forward(self, input_ids, attention_mask):
        # Backbone pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Pooling
        feature = self.pool(last_hidden_state, attention_mask)

        # Classification Head
        if self.config.use_msd and self.training:
            # Multi-Sample Dropout
            logits_list = []
            for dropout in self.dropouts:
                logits_list.append(self.fc(dropout(feature)))
            # Average the logits
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            # Standard Inference (Dropout is identity in eval mode)
            # Or standard dropout if MSD is disabled
            if self.config.use_msd:
                logits = self.fc(feature)
            else:
                logits = self.fc(self.std_dropout(feature))

        return logits

    def load_tapt_weights(self, path):
        """
        Loads weights from the Task-Adaptive Pre-Training stage.
        Handles loading backbone weights from a MaskedLM model.
        """
        print(f"Loading TAPT weights from {path}")
        state_dict = torch.load(path, map_location="cpu")

        # If the checkpoint is a full dict (has 'state_dict' key)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Prepare new state dict for the backbone
        backbone_state_dict = {}

        # Determine the prefix used in the TAPT model (e.g., "deberta." or "roberta.")
        # We assume the TAPT model was a generic AutoModelForMaskedLM
        # For DeBERTa-v3, the prefix inside ForMaskedLM is usually "deberta"

        # Try to detect prefix
        prefix = ""
        keys = list(state_dict.keys())
        if any(k.startswith("deberta.") for k in keys):
            prefix = "deberta."
        elif any(k.startswith("roberta.") for k in keys):
            prefix = "roberta."
        elif any(k.startswith("bert.") for k in keys):
            prefix = "bert."

        for k, v in state_dict.items():
            # If the key starts with the backbone prefix, map it to the backbone
            if prefix and k.startswith(prefix):
                new_key = k[len(prefix) :]  # Remove prefix
                backbone_state_dict[new_key] = v
            # If there is no prefix (e.g. just the backbone was saved), load directly
            elif (
                not prefix and not k.startswith("cls.") and not k.startswith("lm_head.")
            ):
                backbone_state_dict[k] = v

        # Load into backbone
        missing, unexpected = self.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )
        print(
            f"TAPT Weights Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )
