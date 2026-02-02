import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class InsultModel(nn.Module):
    """
    Insult Detection Model based on RoBERTa-Large.

    Architecture:
    1. RoBERTa-Large Backbone
    2. Mean Pooling (averaging non-padding tokens)
    3. Dropout (0.2)
    4. Linear Classification Head
    """

    def __init__(self, model_name_or_path=None, pretrained=True):
        """
        Initialize the model.

        Args:
            model_name_or_path (str, optional): Path to pretrained model or model identifier.
                                                Defaults to Config.model_name.
            pretrained (bool): Whether to load pretrained weights. Defaults to True.
        """
        super(InsultModel, self).__init__()

        # Use provided path or default to Config
        self.model_name = (
            model_name_or_path if model_name_or_path else Config.model_name
        )
        self.config = AutoConfig.from_pretrained(self.model_name)

        # Initialize Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                self.model_name, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Classification Head Components
        self.dropout = nn.Dropout(Config.dropout)
        self.classifier = nn.Linear(self.config.hidden_size, Config.num_classes)

        # Apply layer freezing strategy
        self.freeze_layers()

    def freeze_layers(self):
        """
        Freezes the embeddings and the bottom N encoder layers of the backbone
        as defined in Config.freeze_encoder_layers.
        """
        # Freeze Embeddings
        if Config.freeze_embeddings:
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False

        # Freeze Encoder Layers
        if Config.freeze_encoder_layers > 0:
            # RoBERTa structure: backbone.encoder.layer is a ModuleList
            if hasattr(self.backbone, "encoder") and hasattr(
                self.backbone.encoder, "layer"
            ):
                layers = self.backbone.encoder.layer
                num_layers = len(layers)
                num_freeze = min(Config.freeze_encoder_layers, num_layers)

                for i in range(num_freeze):
                    for param in layers[i].parameters():
                        param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor, optional): Labels (unused in forward, handled by loss fn).

        Returns:
            torch.Tensor: Logits.
        """
        # Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Shape: [Batch, Seq, Hidden]

        # Mean Pooling Strategy
        # 1. Expand attention mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # 2. Sum hidden states of non-padding tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # 3. Count non-padding tokens (clamp to avoid division by zero)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # 4. Compute mean
        mean_embeddings = sum_embeddings / sum_mask

        # Classification Head
        x = self.dropout(mean_embeddings)
        logits = self.classifier(x)

        return logits
