import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.configuration import Config
from library.utilities import get_logger

logger = get_logger("Modeling")


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Dynamically weights the input sequence tokens to produce a single sentence embedding.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        """
        w = self.attention(last_hidden_state)  # (batch_size, seq_len, 1)

        # Mask padding tokens so they don't contribute to the attention score
        if attention_mask is not None:
            # Create a large negative value for padding tokens
            # attention_mask is 1 for tokens, 0 for padding
            extended_mask = (1.0 - attention_mask.unsqueeze(-1)) * -1e9
            w = w + extended_mask

        weights = torch.softmax(w, dim=1)  # (batch_size, seq_len, 1)

        # Weighted sum of hidden states
        feature = torch.sum(
            weights * last_hidden_state, dim=1
        )  # (batch_size, hidden_size)
        return feature


class EssayModel(nn.Module):
    """
    Main Model Architecture.
    Backbone: DeBERTa-v3-Large
    Pooling: Attention Pooling
    Head: Linear Regression
    """

    def __init__(self, checkpoint_path=None, pretrained=True):
        """
        Args:
            checkpoint_path (str, optional): Path to a local checkpoint (e.g., MLM weights).
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(Config.MODEL_BACKBONE)

        # Disable internal dropout for deterministic regression
        self.config.attention_probs_dropout_prob = 0.0
        self.config.hidden_dropout_prob = 0.0

        if pretrained:
            if checkpoint_path:
                logger.info(f"Loading backbone from checkpoint: {checkpoint_path}")
                self.backbone = AutoModel.from_pretrained(
                    checkpoint_path, config=self.config
                )
            else:
                logger.info(f"Loading backbone from HF Hub: {Config.MODEL_BACKBONE}")
                self.backbone = AutoModel.from_pretrained(
                    Config.MODEL_BACKBONE, config=self.config
                )
        else:
            logger.info("Initializing backbone from config (random weights)")
            self.backbone = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing to save VRAM
        if Config.GRADIENT_CHECKPOINTING:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.pooler = AttentionPooling(self.config.hidden_size)
        self.fc = nn.Linear(self.config.hidden_size, Config.NUM_LABELS)

        self._init_weights(self.pooler)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        Returns logits (scores).
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        feature = self.pooler(last_hidden_state, attention_mask)
        logits = self.fc(feature)

        return logits


def get_optimizer_params(model):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Strategy:
    - Head parameters (Pooler + FC) get Config.HEAD_LEARNING_RATE.
    - Backbone layers get decayed LR based on depth: LR * (decay ^ depth).
    - Embeddings get the lowest LR.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = {}

    num_layers = model.config.num_hidden_layers

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 1. Determine Weight Decay
        if any(nd in n for nd in no_decay):
            wd = 0.0
        else:
            wd = Config.WEIGHT_DECAY

        # 2. Determine Learning Rate
        if "backbone" not in n:
            # Head parameters
            lr = Config.HEAD_LEARNING_RATE
        else:
            # Backbone parameters
            # Identify layer index
            layer_idx = -1

            if "encoder.layer." in n:
                try:
                    # Example name: backbone.encoder.layer.15.output.dense.weight
                    # Split to isolate the layer number
                    parts = n.split("encoder.layer.")[1].split(".")
                    layer_idx = int(parts[0])
                except (IndexError, ValueError):
                    layer_idx = -1

            if layer_idx != -1:
                # Calculate decay based on distance from top
                # Top layer (num_layers - 1) -> decay^0
                # Bottom layer (0) -> decay^(num_layers - 1)
                distance_from_top = (num_layers - 1) - layer_idx
                lr = Config.LEARNING_RATE * (Config.LLRD_DECAY**distance_from_top)
            else:
                # Embeddings or other non-layer params (deepest)
                lr = Config.LEARNING_RATE * (Config.LLRD_DECAY**num_layers)

        # 3. Group parameters
        # We use (lr, wd) as the key to group parameters
        key = (lr, wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(p)

    # Convert groups to list of dicts format required by optimizer
    optimizer_parameters = []
    for (lr, wd), params in param_groups.items():
        optimizer_parameters.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_parameters
