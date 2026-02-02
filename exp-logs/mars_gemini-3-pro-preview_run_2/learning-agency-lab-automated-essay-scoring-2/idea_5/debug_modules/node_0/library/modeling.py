import torch
import torch.nn as nn
import numpy as np
import time
from transformers import AutoConfig, AutoModel
from library.config import Config
from library.utils import compute_qwk


# =========================================================================================
# Pooling Layer
# =========================================================================================
class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Max Pooling
        # Create a temporary tensor for max pooling to avoid modifying last_hidden_state in-place
        # We set padding tokens to a very large negative value so they are not selected by max()
        # (1 - input_mask_expanded) is 1 where padding exists, 0 otherwise.
        max_embeddings = torch.max(
            last_hidden_state * input_mask_expanded + (1 - input_mask_expanded) * -1e9,
            1,
        )[0]

        # Concatenate mean and max embeddings
        return torch.cat([mean_embeddings, max_embeddings], 1)


# =========================================================================================
# Model Architecture
# =========================================================================================
class DebertaRegressor(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        super(DebertaRegressor, self).__init__()

        # Load Config
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.config.hidden_dropout_prob = Config.HIDDEN_DROPOUT
        self.config.attention_probs_dropout_prob = Config.ATTENTION_DROPOUT

        # Load Model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Pooling and Head
        self.pooling = MeanMaxPooling()
        # Input to FC is hidden_size * 2 because of Mean+Max pooling
        self.fc = nn.Linear(self.config.hidden_size * 2, Config.NUM_LABELS)

        # Initialize Head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pooling(last_hidden_state, attention_mask)
        logits = self.fc(feature)
        return logits


# =========================================================================================
# Adversarial Weight Perturbation (AWP)
# =========================================================================================
class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.AWP_ADV_LR,
        adv_eps=Config.AWP_ADV_EPS,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        e = 1e-6
        self._save()  # Save current weights
        for name, param in self.model.named_parameters():
            # Apply perturbation only to weights (not bias/norm) and if they have gradients
            if param.grad is not None and self.adv_param in name:
                norm = torch.norm(param.grad)
                # Avoid division by zero or perturbing with NaNs
                if norm > 0 and not torch.isnan(norm) and not torch.isinf(norm):
                    # Compute perturbation: r_at = adv_lr * grad / norm
                    # Note: If gradients are scaled (AMP), both grad and norm are scaled, so the ratio is correct.
                    r_at = self.adv_lr * param.grad / (norm + e)
                    param.data.add_(r_at)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =========================================================================================
# Training Logic
# =========================================================================================
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, awp=None):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    # Loss Function
    criterion = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_BETA)

    # Gradient Accumulation setup
    scaler = torch.cuda.amp.GradScaler()

    start = time.time()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["score"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), labels)
            loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward (Accumulate clean gradients)
        scaler.scale(loss).backward()

        # AWP Logic & Optimizer Step
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:

            # Apply AWP if enabled and epoch condition met
            if Config.USE_AWP and awp is not None and epoch >= Config.AWP_START_EPOCH:
                # We attack using the accumulated gradients.
                # Note: We do not unscale here to avoid mixing unscaled/scaled states in complex ways.
                # The direction of scaled gradients is valid for AWP perturbation.
                awp.attack()

                # Forward pass with perturbed weights
                with torch.cuda.amp.autocast():
                    outputs_adv = model(input_ids, attention_mask)
                    loss_adv = criterion(outputs_adv.view(-1), labels)
                    loss_adv = loss_adv / Config.GRAD_ACCUM_STEPS

                # Backward pass for adversarial loss (Accumulate adversarial gradients)
                scaler.scale(loss_adv).backward()

                # Restore original weights
                awp.restore()

            # Update weights
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * Config.GRAD_ACCUM_STEPS) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size

    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    preds = []
    labels_list = []

    criterion = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_BETA)

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["score"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions for QWK calculation
            preds.append(outputs.view(-1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    preds = np.concatenate(preds)
    labels_list = np.concatenate(labels_list)

    # Compute QWK
    val_qwk = compute_qwk(labels_list, preds)

    return epoch_loss, val_qwk, preds
