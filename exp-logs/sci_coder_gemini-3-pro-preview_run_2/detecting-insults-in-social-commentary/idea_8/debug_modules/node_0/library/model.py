import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import os


class InsultModel(nn.Module):
    """
    Insult Detection Model based on DeBERTa-v3-Large.
    Supports loading from Task-Adaptive Pre-Training (TAPT) checkpoints
    and implements Multi-Sample Dropout (MSD) for robust classification.
    """

    def __init__(self, config, model_path=None, pretrained=True):
        super().__init__()
        self.config = config

        # Logic to determine the correct model path
        # Priority: Explicit argument > TAPT output directory > Config model name
        if model_path is None:
            tapt_path = config.tapt_output_dir
            # Check for essential HF model files to confirm validity
            has_config = os.path.exists(os.path.join(tapt_path, "config.json"))
            has_weights = os.path.exists(
                os.path.join(tapt_path, "pytorch_model.bin")
            ) or os.path.exists(os.path.join(tapt_path, "model.safetensors"))

            if has_config and has_weights:
                model_path = tapt_path
                print(
                    f"InsultModel: Initializing backbone from TAPT checkpoint: {model_path}"
                )
            else:
                model_path = config.model_name
                print(
                    f"InsultModel: Initializing backbone from base model: {model_path}"
                )

        # Load Configuration
        self.model_config = AutoConfig.from_pretrained(model_path)
        # Disable internal dropout to rely on MSD head
        self.model_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
            }
        )

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_path, config=self.model_config
            )
        else:
            self.backbone = AutoModel.from_config(self.model_config)

        # Enable Gradient Checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # Head Configuration
        self.hidden_size = self.model_config.hidden_size
        self.pooler_type = config.pooler_type

        # Multi-Sample Dropout Setup
        self.use_msd = config.use_msd
        self.msd_num = config.msd_num if self.use_msd else 1

        # Create multiple dropout layers for MSD
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.fc_dropout) for _ in range(self.msd_num)]
        )

        # Final Classification Layer
        self.fc = nn.Linear(self.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize the weights of the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        """
        Extracts pooled features from the backbone.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if self.pooler_type == "mean":
            # Mean Pooling with Attention Mask
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
            feature = sum_embeddings / sum_mask
        else:
            # Fallback to CLS token
            feature = last_hidden_state[:, 0, :]

        return feature

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        Applies Multi-Sample Dropout during training.
        """
        feature = self.feature(input_ids, attention_mask)

        if self.use_msd and self.training:
            # Multi-Sample Dropout: Average the logits from multiple dropout masks
            logits_list = []
            for dropout in self.dropouts:
                logits_list.append(self.fc(dropout(feature)))

            # Stack and mean to get the ensemble prediction
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            # Inference or MSD disabled: Single pass
            # Note: During eval, nn.Dropout is an identity op
            logits = self.fc(feature)

        return logits
