import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class InsultModel(nn.Module):
    """
    InsultModel based on DeBERTa-v3-Large with Mean Pooling and specific freezing strategy.
    """

    def __init__(self, pretrained_model_name=Config.MODEL_NAME):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # ==========================================
        # Freezing Strategy
        # ==========================================
        # 1. Freeze Embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # 2. Freeze Bottom N Encoder Layers
        # DeBERTa-v3 structure: backbone.encoder.layer is a ModuleList
        if hasattr(Config, "FREEZE_LAYERS") and Config.FREEZE_LAYERS > 0:
            # Ensure we don't try to freeze more layers than exist
            num_layers_to_freeze = min(
                Config.FREEZE_LAYERS, len(self.backbone.encoder.layer)
            )
            for i in range(num_layers_to_freeze):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

        # ==========================================
        # Classification Head
        # ==========================================
        self.fc = nn.Linear(self.config.hidden_size, Config.NUM_CLASSES)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the classification head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass with Mean Pooling.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits.
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # ==========================================
        # Mean Pooling
        # ==========================================
        # Expand attention mask to match hidden state dimensions: (Batch, SeqLen, Hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum hidden states masked by attention mask
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum the mask to get the count of valid tokens (avoid div by zero)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask

        # ==========================================
        # Classification
        # ==========================================
        logits = self.fc(mean_embeddings)

        return logits
