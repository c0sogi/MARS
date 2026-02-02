import torch
import torch.nn as nn
import numpy as np
import time
from library.config import CFG
from library.utils import AverageMeter, get_score


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def attack(self, epoch):
        if epoch < self.start_epoch:
            return
        self._save()
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
                    # Compute perturbation: direction of gradient * learning rate
                    # Normalized by gradient norm to get direction, scaled by weight norm
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

    def restore(self, epoch):
        if epoch < self.start_epoch:
            return
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    # Determine number of layers (DeBERTa-v3-Large typically has 24)
    num_layers = 24
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers

    # Group parameters by layer index
    # 0: Embeddings
    # 1..num_layers: Encoder Layers (1 to 24)
    # num_layers + 1: Head / Custom Layers
    groups = {i: [] for i in range(num_layers + 2)}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "embeddings" in name:
            layer_id = 0
        elif "encoder.layer" in name:
            # Extract layer index from name (e.g., model.encoder.layer.11.output...)
            try:
                parts = name.split(".")
                idx = -1
                for part in parts:
                    if part.isdigit():
                        idx = int(part)
                        break
                if idx != -1:
                    layer_id = idx + 1
                else:
                    layer_id = 0
            except:
                layer_id = 0
        else:
            # Heads, pooler, mixing layers
            layer_id = num_layers + 1

        groups[layer_id].append((name, p))

    optimizer_parameters = []
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    for layer_id, params in groups.items():
        if not params:
            continue

        if layer_id == num_layers + 1:
            lr = decoder_lr
        else:
            # LLRD: lr = encoder_lr * (decay ** distance_from_top)
            if layer_id == 0:
                distance = num_layers
            else:
                distance = num_layers - layer_id

            lr = encoder_lr * (CFG.llrd_decay**distance)

        decay_params = []
        no_decay_params = []

        for n, p in params:
            if any(nd in n for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if decay_params:
            optimizer_parameters.append(
                {"params": decay_params, "weight_decay": weight_decay, "lr": lr}
            )
        if no_decay_params:
            optimizer_parameters.append(
                {"params": no_decay_params, "weight_decay": 0.0, "lr": lr}
            )

    return optimizer_parameters


def train_fn(
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    awp=None,
    ema=None,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = AverageMeter()

    for step, inputs in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["labels"]
        batch_size = labels.size(0)

        # 1. Forward Pass (Clean)
        with torch.cuda.amp.autocast():
            outputs = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss_dict = criterion(outputs, labels)
            loss = loss_dict["loss"]

        losses.update(loss.item(), batch_size)

        # 2. Backward Pass (Clean)
        scaler.scale(loss).backward()

        # 3. AWP Attack & Backward
        if awp is not None and epoch >= awp.start_epoch:
            # Perturb weights based on current gradients
            awp.attack(epoch)

            # Clear gradients to step only on the adversarial direction
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs_adv = model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs.get("token_type_ids"),
                )
                loss_dict_adv = criterion(outputs_adv, labels)
                loss_adv = loss_dict_adv["loss"]

            # Accumulate adversarial gradients
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore(epoch)

        # 4. Optimization Step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update()

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Grad: {grad_norm:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, ema=None):
    # Apply EMA weights for validation if available
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    for step, inputs in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["labels"]
        batch_size = labels.size(0)

        with torch.no_grad():
            outputs = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss_dict = criterion(outputs, labels)
            loss = loss_dict["loss"]

        losses.update(loss.item(), batch_size)

        # Collect score predictions
        preds.append(outputs["score"].view(-1).cpu().numpy())
        targets.append(labels.view(-1).cpu().numpy())

    # Restore original weights after validation
    if ema is not None:
        ema.restore()

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    # Calculate Pearson Correlation
    score = get_score(ground_truth, predictions)

    return losses.avg, score, predictions
