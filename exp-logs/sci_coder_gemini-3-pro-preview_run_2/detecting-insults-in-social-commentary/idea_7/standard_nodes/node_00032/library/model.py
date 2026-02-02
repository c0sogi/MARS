import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from library.config import Config
from library.utils import get_logger

logger = get_logger()


class InsultModel(nn.Module):
    """
    The main classification model based on DeBERTa-v3-large.
    Features:
    - Mean Pooling of the last hidden state.
    - Multi-Sample Dropout (MSD) for regularization.
    - Binary classification head.
    """

    def __init__(self, model_path=None, pretrained=True):
        super().__init__()

        # Determine which model path to use (default from Config or provided override)
        self.model_path = model_path if model_path else Config.model_name

        logger.info(f"Initializing InsultModel with backbone: {self.model_path}")

        # Load Configuration
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,  # We handle dropout in the head
                "attention_probs_dropout_prob": 0.0,  # Often reduced for fine-tuning stability
                "num_labels": Config.num_classes,
            }
        )

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                self.model_path, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Classification Head
        self.use_msd = Config.use_multi_sample_dropout

        # Multi-Sample Dropout: 5 distinct dropout layers
        if self.use_msd:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(Config.fc_dropout) for _ in range(5)]
            )
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(Config.fc_dropout)])

        self.fc = nn.Linear(self.config.hidden_size, Config.num_classes)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
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

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Aggregates the token embeddings using the attention mask to ignore padding.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        """
        # 1. Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Pooling
        feature = self.mean_pooling(last_hidden_state, attention_mask)

        # 3. Head (Multi-Sample Dropout)
        if self.use_msd and self.training:
            # During training with MSD, average the logits from multiple dropout passes
            logits_list = []
            for dropout in self.dropouts:
                logits_list.append(self.fc(dropout(feature)))
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            # During inference or if MSD is disabled, use the first dropout (no-op in eval)
            logits = self.fc(self.dropouts[0](feature))

        return logits
