import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling: Aggregates the last 'num_layers' hidden states
    using learnable scalar weights.
    """

    def __init__(self, num_layers=4):
        super(WeightedLayerPooling, self).__init__()
        self.num_layers = num_layers
        # Initialize weights to 1/num_layers (uniform) initially, but let them learn
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensor (batch, seq_len, hidden_size)
        # We take the last 'num_layers'
        selected_layers = all_hidden_states[-self.num_layers :]

        # Stack them: (num_layers, batch, seq_len, hidden_size)
        stacked = torch.stack(selected_layers)

        # Compute softmax weights: (num_layers, 1, 1, 1)
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum
        output = (w * stacked).sum(dim=0)
        return output


class SharedGatedConvHead(nn.Module):
    """
    Shared Gated Convolutional Head.
    Applies a Gated Linear Unit (GLU) mechanism via 1D Convolutions,
    followed by a shared linear projection for Start and End logits.
    """

    def __init__(self, hidden_size, dropout_prob=0.1):
        super(SharedGatedConvHead, self).__init__()

        # Parallel 1D Convolutions
        # Input: (Batch, Hidden, Seq) -> Output: (Batch, Hidden, Seq)
        self.conv_content = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )
        self.conv_gate = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout_prob)

        # Projection to 2 logits (Start, End)
        self.classifier = nn.Linear(hidden_size, 2)

        # Initialization
        nn.init.xavier_uniform_(self.conv_content.weight)
        nn.init.xavier_uniform_(self.conv_gate.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)

        # Permute for Conv1d: (batch, hidden_size, seq_len)
        x_perm = x.permute(0, 2, 1)

        # Content Stream
        c = self.conv_content(x_perm)

        # Gate Stream
        g = torch.sigmoid(self.conv_gate(x_perm))

        # Gated Output (Element-wise product)
        h = c * g

        # Permute back: (batch, seq_len, hidden_size)
        h = h.permute(0, 2, 1)

        # Dropout
        h = self.dropout(h)

        # Projection
        logits = self.classifier(h)

        return logits


class TweetModel(nn.Module):
    """
    Main Model Class.
    Backbone: DeBERTa-v3-Large
    Pooling: WeightedLayerPooling
    Head: SharedGatedConvHead
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load Config and Backbone
        # We need output_hidden_states=True for the pooling layer
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_PATH, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(Config.MODEL_PATH, config=self.config)

        # Pooling Layer (Last 4 layers)
        self.pooling = WeightedLayerPooling(num_layers=4)

        # Head
        self.head = SharedGatedConvHead(self.config.hidden_size, Config.HEAD_DROPOUT)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Forward pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract hidden states
        all_hidden_states = outputs.hidden_states

        # Pooling
        feature = self.pooling(all_hidden_states)

        # Head
        logits = self.head(feature)

        # Split logits into Start and End
        # logits shape: (batch, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze last dim: (batch, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


def get_optimizer_params(model, learning_rate, weight_decay, llrd_decay):
    """
    Constructs optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD).
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    # Helper to determine parameter group
    param_config = {}
    num_layers = model.config.num_hidden_layers

    for n, p in model.named_parameters():
        # Default values
        lr = learning_rate
        wd = weight_decay

        # 1. Determine Learning Rate based on depth
        if "backbone" not in n:
            # Head and Pooling layers get the base Learning Rate
            lr = learning_rate
        else:
            # Backbone parameters
            if "embeddings" in n:
                # Embeddings get the lowest LR (decayed N+1 times)
                lr = learning_rate * (llrd_decay ** (num_layers + 1))
            elif "encoder" in n:
                # Encoder layers
                if "layer." in n:
                    # Extract layer index
                    # format: backbone.encoder.layer.{i}. ...
                    try:
                        parts = n.split(".")
                        layer_idx_pos = parts.index("layer") + 1
                        layer_idx = int(parts[layer_idx_pos])

                        # Top layer (N-1) gets decay^1
                        # Bottom layer (0) gets decay^N
                        decay_power = num_layers - layer_idx
                        lr = learning_rate * (llrd_decay**decay_power)
                    except ValueError:
                        # Fallback for encoder params not strictly inside 'layer' block
                        lr = learning_rate * llrd_decay
                else:
                    # Other encoder params (e.g. relative embeddings)
                    lr = learning_rate * (llrd_decay ** (num_layers + 1))
            else:
                # Catch-all for other backbone params
                lr = learning_rate * (llrd_decay ** (num_layers + 1))

        # 2. Determine Weight Decay
        if any(nd in n for nd in no_decay):
            wd = 0.0

        # Store config
        param_config[n] = {"lr": lr, "weight_decay": wd, "param": p}

    # Group parameters by (lr, weight_decay)
    # We sort by name to ensure deterministic order
    sorted_names = sorted(param_config.keys())

    groups = {}  # Key: (lr, wd) -> list of params

    for n in sorted_names:
        cfg = param_config[n]
        key = (cfg["lr"], cfg["weight_decay"])
        if key not in groups:
            groups[key] = []
        groups[key].append(cfg["param"])

    # Create final list for optimizer
    for (lr, wd), params in groups.items():
        optimizer_grouped_parameters.append(
            {"params": params, "weight_decay": wd, "lr": lr}
        )

    return optimizer_grouped_parameters
