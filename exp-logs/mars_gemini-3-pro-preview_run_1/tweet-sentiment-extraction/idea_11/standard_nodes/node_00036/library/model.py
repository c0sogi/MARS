import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Aggregates the last N hidden layers using learnable weights.
    """

    def __init__(self):
        super(WeightedLayerPooling, self).__init__()
        self.num_pooling_layers = Config.N_POOLING_LAYERS
        # Initialize weights to be equal (softmax will make them uniform initially)
        self.layer_weights = nn.Parameter(torch.tensor([1.0] * self.num_pooling_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of (embeddings, layer_0, ..., layer_23)
        # Select the last N layers
        selected_layers = all_hidden_states[-self.num_pooling_layers :]

        # Stack to shape: (Batch, Seq, Hidden, N_Layers)
        stacked = torch.stack(selected_layers, dim=-1)

        # Compute softmax over the layer weights
        weights = torch.softmax(self.layer_weights, dim=0)

        # Weighted sum: (Batch, Seq, Hidden, N) * (1, 1, 1, N) -> Sum over last dim
        weighted_sum = (stacked * weights.view(1, 1, 1, -1)).sum(dim=-1)

        return weighted_sum


class SentimentModel(nn.Module):
    """
    DeBERTa-v3-Large with Weighted Layer Pooling and CNN Head.
    """

    def __init__(self):
        super(SentimentModel, self).__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.output_hidden_states = True

        # Backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Head Components
        self.pooling = WeightedLayerPooling()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.config.hidden_size,
                out_channels=Config.CNN_OUT_CHANNELS,
                kernel_size=Config.CNN_KERNEL_SIZE,
                padding=(Config.CNN_KERNEL_SIZE - 1) // 2,
            ),
            nn.GELU(),
        )

        self.fc = nn.Linear(Config.CNN_OUT_CHANNELS, 2)

        # Initialize Head Weights
        self._init_weights(self.cnn)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Weighted Pooling of last N layers
        feature = self.pooling(outputs.hidden_states)

        # CNN Head
        # Conv1d expects (Batch, Channels, Seq)
        feature = feature.permute(0, 2, 1)
        feature = self.cnn(feature)

        # Permute back to (Batch, Seq, Channels) for Linear
        feature = feature.permute(0, 2, 1)

        # Final Projection
        logits = self.fc(feature)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        return start_logits.squeeze(-1), end_logits.squeeze(-1)

    def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay=0.0):
        """
        Configures parameters for the optimizer with Layer-wise Learning Rate Decay (LLRD).
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = []

        # 1. Head Parameters (Highest LR)
        head_names = ["pooling", "cnn", "fc"]
        head_params_decay = [
            p
            for n, p in param_optimizer
            if any(hn in n for hn in head_names) and not any(nd in n for nd in no_decay)
        ]
        head_params_no_decay = [
            p
            for n, p in param_optimizer
            if any(hn in n for hn in head_names) and any(nd in n for nd in no_decay)
        ]

        optimizer_parameters.append(
            {
                "params": head_params_decay,
                "weight_decay": weight_decay,
                "lr": decoder_lr,
            }
        )
        optimizer_parameters.append(
            {"params": head_params_no_decay, "weight_decay": 0.0, "lr": decoder_lr}
        )

        # 2. Backbone Layers (Decaying LR)
        n_layers = self.config.num_hidden_layers

        # Iterate from top layer (closest to head) to bottom layer
        for i in range(n_layers - 1, -1, -1):
            # Calculate decayed LR
            layer_lr = encoder_lr * (Config.LLRD_DECAY ** (n_layers - 1 - i))
            layer_name = f"encoder.layer.{i}."

            layer_params_decay = [
                p
                for n, p in param_optimizer
                if layer_name in n and not any(nd in n for nd in no_decay)
            ]
            layer_params_no_decay = [
                p
                for n, p in param_optimizer
                if layer_name in n and any(nd in n for nd in no_decay)
            ]

            optimizer_parameters.append(
                {
                    "params": layer_params_decay,
                    "weight_decay": weight_decay,
                    "lr": layer_lr,
                }
            )
            optimizer_parameters.append(
                {"params": layer_params_no_decay, "weight_decay": 0.0, "lr": layer_lr}
            )

        # 3. Embeddings and other bottom-level params (Lowest LR)
        embed_lr = encoder_lr * (Config.LLRD_DECAY**n_layers)

        # Identify parameters already assigned
        assigned_ids = set(
            id(p) for group in optimizer_parameters for p in group["params"]
        )

        # Assign remaining parameters (embeddings, relative positions, etc.)
        rest_params_decay = [
            p
            for n, p in param_optimizer
            if id(p) not in assigned_ids and not any(nd in n for nd in no_decay)
        ]
        rest_params_no_decay = [
            p
            for n, p in param_optimizer
            if id(p) not in assigned_ids and any(nd in n for nd in no_decay)
        ]

        optimizer_parameters.append(
            {"params": rest_params_decay, "weight_decay": weight_decay, "lr": embed_lr}
        )
        optimizer_parameters.append(
            {"params": rest_params_no_decay, "weight_decay": 0.0, "lr": embed_lr}
        )

        return optimizer_parameters
