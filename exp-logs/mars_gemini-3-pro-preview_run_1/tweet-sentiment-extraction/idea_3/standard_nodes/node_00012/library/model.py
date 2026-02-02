import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from library.config import TweetConfig


class TweetModel(nn.Module):
    """
    Neural Network for Tweet Sentiment Extraction.
    Backbone: DeBERTa-v3-base
    Head: Weighted Layer Pooling + 1D Convolution + Linear
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.config = TweetConfig()

        # Load HuggingFace Configuration
        # We need output_hidden_states=True for the weighted layer pooling
        hf_config = AutoConfig.from_pretrained(
            self.config.MODEL_NAME, output_hidden_states=True
        )

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                self.config.MODEL_NAME, config=hf_config
            )
        else:
            self.backbone = AutoModel.from_config(hf_config)

        # Weighted Layer Pooling Weights
        # Learnable weights for the last 4 layers, initialized uniformly
        self.layer_weights = nn.Parameter(torch.tensor([1 / 4] * 4, dtype=torch.float))

        # CNN-Enhanced Head
        # Conv1d to capture local n-gram context
        # Input/Output channels = Hidden Size (768)
        self.conv1d = nn.Conv1d(
            in_channels=self.config.HIDDEN_SIZE,
            out_channels=self.config.HIDDEN_SIZE,
            kernel_size=3,
            padding=1,
        )

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in self.config.MS_DROPOUT_RATES]
        )

        # Final Linear Layer to predict start and end logits
        self.fc = nn.Linear(self.config.HIDDEN_SIZE, 2)

        # Initialize custom head weights
        self._init_weights(self.conv1d)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head components.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.HIDDEN_SIZE**-0.5)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (Batch, Seq_Len)
            attention_mask (torch.Tensor): Attention mask (Batch, Seq_Len)

        Returns:
            start_logits (torch.Tensor): Logits for start index (Batch, Seq_Len)
            end_logits (torch.Tensor): Logits for end index (Batch, Seq_Len)
        """
        # Pass through backbone
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        # Extract hidden states
        # outputs.hidden_states is a tuple of tensors, each (Batch, Seq_Len, Hidden)
        all_hidden_states = outputs.hidden_states

        # Stack the last 4 layers -> (4, Batch, Seq_Len, Hidden)
        stack = torch.stack(all_hidden_states[-4:])

        # Calculate Softmax weights for pooling -> (4, 1, 1, 1)
        weights = F.softmax(self.layer_weights, dim=0).view(4, 1, 1, 1)

        # Weighted Sum Pooling -> (Batch, Seq_Len, Hidden)
        weighted_output = torch.sum(weights * stack, dim=0)

        # Prepare for Conv1d: Permute to (Batch, Hidden, Seq_Len)
        x = weighted_output.permute(0, 2, 1)

        # Apply 1D Convolution and Activation
        x = self.conv1d(x)
        x = F.relu(x)

        # Permute back to (Batch, Seq_Len, Hidden)
        x = x.permute(0, 2, 1)

        # Multi-Sample Dropout and Linear Projection
        logits = None
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.fc(dropout(x))
            else:
                logits += self.fc(dropout(x))

        logits = logits / len(self.dropouts)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (Batch, Seq_Len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
