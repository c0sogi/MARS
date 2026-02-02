import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class SentimentModel(nn.Module):
    """
    The SentimentModel architecture consisting of:
    1. DeBERTa-v3 Backbone
    2. Weighted Layer Pooling (last 4 layers)
    3. Bi-Directional LSTM
    4. 1D Convolutional Context Layer
    5. Linear Prediction Head
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 1. Backbone Configuration
        model_config = AutoConfig.from_pretrained(config.model_name)
        model_config.output_hidden_states = True

        # Load Pretrained Model
        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=model_config
        )

        # 2. Weighted Layer Pooling
        self.n_pooling_layers = config.n_pooling_layers
        # Learnable weights for the last n layers
        self.layer_weights = nn.Parameter(
            torch.ones(self.n_pooling_layers) / self.n_pooling_layers
        )

        # 3. Bi-Directional LSTM
        # Input size is hidden_size (768).
        # Hidden size is lstm_hidden_size (384).
        # Bidirectional=True results in output size 384*2 = 768.
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # 4. CNN Context Layer
        # Input channels: 768 (from BiLSTM)
        # Output channels: 768
        # Kernel size: 3
        self.cnn = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.cnn_out_channels,
            kernel_size=config.cnn_kernel_size,
            padding=(config.cnn_kernel_size - 1) // 2,
        )
        self.act = nn.GELU()

        # 5. Output Head
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.cnn_out_channels, 2)

        # Initialize custom layers
        self._init_weights(self.lstm)
        self._init_weights(self.cnn)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for custom layers (LSTM, CNN, Linear).
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.hidden_size**-0.5)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        """
        Extracts features using Backbone + Weighted Layer Pooling.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple of (layer_0, ..., layer_N)
        # We take the last n_pooling_layers
        all_hidden_states = outputs.hidden_states

        # Stack: (batch, n_pooling_layers, seq_len, hidden_size)
        stack = torch.stack(all_hidden_states[-self.n_pooling_layers :], dim=1)

        # Calculate softmax weights: (n_pooling_layers)
        weights = torch.softmax(self.layer_weights, dim=0)

        # Reshape for broadcasting: (1, n_pooling_layers, 1, 1)
        weights = weights.view(1, -1, 1, 1)

        # Weighted sum: (batch, seq_len, hidden_size)
        weighted_sum = torch.sum(weights * stack, dim=1)

        return weighted_sum

    def forward(self, input_ids, attention_mask):
        # 1. Feature Extraction
        x = self.feature(input_ids, attention_mask)  # (batch, seq_len, 768)

        # 2. LSTM
        # Output: (batch, seq_len, 2 * lstm_hidden_size) -> (batch, seq_len, 768)
        self.lstm.flatten_parameters()  # Optimization for RNNs
        x, _ = self.lstm(x)

        # 3. CNN
        # Permute for Conv1d: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.act(x)
        # Permute back: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        # 4. Classification Head
        x = self.dropout(x)
        logits = self.fc(x)  # (batch, seq_len, 2)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch, seq_len)
        end_logits = end_logits.squeeze(-1)  # (batch, seq_len)

        return start_logits, end_logits
