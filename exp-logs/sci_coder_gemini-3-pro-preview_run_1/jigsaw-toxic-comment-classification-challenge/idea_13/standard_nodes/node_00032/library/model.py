import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs the model weights to maximize the loss during training,
    encouraging the model to find flatter, more robust minima.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.2):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class DeepSupervisedModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            Config.model_name, output_hidden_states=True
        )

        if pretrained:
            self.model = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.model = AutoModel.from_config(self.config)

        # --- Deep Supervision Head (Layer 6) ---
        # Global Max Pooling -> Dense
        self.aux_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, Config.num_classes)
        )

        # --- Main Head Aggregation ---
        # Learnable weights for the last N layers
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * Config.num_last_layers_agg, dtype=torch.float)
        )

        # --- Hybrid Pooling ---
        # 1. Linear Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )

        # 2. Global Max Pooling (No parameters needed, just logic in forward)

        # --- Multi-Sample Dropout & Classifier ---
        # We use 5 dropout masks
        self.dropouts = nn.ModuleList([nn.Dropout(Config.fc_dropout) for _ in range(5)])

        # Input dim is hidden_size * 2 because of Hybrid Pooling (Attention + Max)
        self.fc = nn.Linear(self.config.hidden_size * 2, Config.num_classes)

        # Initialize custom layers
        self._init_weights(self.aux_head)
        self._init_weights(self.attention)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature_extraction(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

        # --- Aux Feature (Layer 6) ---
        # hidden_states[0] is embeddings, so index 6 corresponds to the output of the 6th encoder layer
        aux_hidden = all_hidden_states[Config.deep_supervision_layer]

        # Mask padding tokens for Max Pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(aux_hidden.size()).float()
        )
        aux_hidden_masked = aux_hidden.clone()
        aux_hidden_masked[input_mask_expanded == 0] = -1e9

        # Global Max Pooling for Aux Head
        aux_feature = torch.max(aux_hidden_masked, 1)[0]

        # --- Main Feature (Weighted Last N Layers) ---
        # Get last N layers
        last_n_hidden = torch.stack(
            all_hidden_states[-Config.num_last_layers_agg :]
        )  # (N, batch, seq, hidden)

        # Weighted sum
        weights = F.softmax(self.layer_weights, dim=0)
        weights = weights.view(-1, 1, 1, 1)  # Reshape for broadcasting
        main_hidden = (weights * last_n_hidden).sum(dim=0)  # (batch, seq, hidden)

        # --- Hybrid Pooling ---
        # 1. Linear Attention Pooling
        att_weights = self.attention(main_hidden)  # (batch, seq, 1)
        context_vector = torch.sum(att_weights * main_hidden, dim=1)  # (batch, hidden)

        # 2. Global Max Pooling on Main Hidden
        main_hidden_masked = main_hidden.clone()
        main_hidden_masked[input_mask_expanded == 0] = -1e9
        max_vector = torch.max(main_hidden_masked, 1)[0]  # (batch, hidden)

        # Concatenate: (batch, hidden*2)
        final_feature = torch.cat([context_vector, max_vector], dim=1)

        return aux_feature, final_feature

    def forward(self, input_ids, attention_mask, labels=None):
        aux_feature, final_feature = self.feature_extraction(input_ids, attention_mask)

        # Aux Logits
        aux_logits = self.aux_head(aux_feature)

        # Main Logits with Multi-Sample Dropout
        # Pass feature through each dropout mask, then through FC, then average
        main_logits = torch.mean(
            torch.stack(
                [self.fc(dropout(final_feature)) for dropout in self.dropouts], dim=0
            ),
            dim=0,
        )

        return {"main_logits": main_logits, "aux_logits": aux_logits}
