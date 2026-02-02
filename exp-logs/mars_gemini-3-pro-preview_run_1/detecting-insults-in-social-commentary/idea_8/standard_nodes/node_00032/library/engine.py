import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from library.utils import AverageMeter
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs the model weights to maximize the loss, effectively flattening the loss landscape
    and improving generalization.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.2):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(
        self, inputs, labels, attention_mask, struct_features, criterion
    ):
        """
        Performs the attack on the model weights and computes the gradient on the perturbed weights.
        """
        with torch.cuda.amp.autocast(enabled=False):
            self._save()
            self._attack_step()

            # Forward pass with perturbed weights
            y_preds = self.model(inputs, attention_mask, struct_features)
            adv_loss = criterion(y_preds.view(-1), labels.view(-1))

            # Zero optimizer gradients before backward on perturbed loss?
            # Standard AWP usually accumulates gradients or replaces them.
            # Here we follow the strategy: backward() on original -> attack -> backward() on perturbed -> restore.
            # But PyTorch accumulates by default.
            self.optimizer.zero_grad()
            adv_loss.backward()

            self._restore()

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

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def train_fn(
    dataloader, model, criterion, optimizer, epoch, scheduler, device, config=Config
):
    """
    Training loop for one epoch.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = AverageMeter()

    # Initialize AWP if enabled
    awp = None
    if config.use_awp and epoch >= config.awp_start_epoch:
        awp = AWP(model, optimizer, adv_lr=config.awp_lr, adv_eps=config.awp_eps)

    # Progress bar
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Train Epoch {epoch}")

    for step, data in enumerate(pbar):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        struct_features = data["struct_features"].to(device)
        labels = data["label"].to(device)

        batch_size = input_ids.size(0)

        # Standard Forward Pass
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(input_ids, attention_mask, struct_features)
            loss = criterion(y_preds.view(-1), labels.view(-1))

        # Accumulate loss for metrics
        losses.update(loss.item(), batch_size)

        # Standard Backward
        scaler.scale(loss).backward()

        # AWP Attack (if active)
        if awp is not None:
            # Unscale gradients to allow AWP to access valid gradients
            scaler.unscale_(optimizer)

            # Perform attack: save -> perturb -> forward -> backward -> restore
            # We manually implement the logic here to integrate with scaler
            awp._save()
            awp._attack_step()

            with torch.cuda.amp.autocast(enabled=True):
                y_preds_adv = model(input_ids, attention_mask, struct_features)
                loss_adv = criterion(y_preds_adv.view(-1), labels.view(-1))

            # We don't zero grad here; we want to accumulate or replace.
            # In AWP usually we want the gradient direction of the perturbed landscape.
            # A common robust strategy is to zero grads from the clean pass and only step with adv grads,
            # or weight them. Here we follow a standard implementation:
            # Since we already called backward on clean loss, we have clean gradients.
            # To strictly follow AWP, we often clear clean gradients and use adv gradients,
            # or use a mix. Given the complexity, we will use the standard AWP approach:
            # 1. Clean Backward (done)
            # 2. Perturb weights based on clean grads
            # 3. Adv Forward & Backward (accumulating to clean grads or replacing)
            # For simplicity and stability in this pipeline, we will use the clean gradients
            # to guide the perturbation, but we won't do a second backward pass to keep runtime low
            # unless explicitly required. However, AWP *requires* the gradient at the perturbed point.

            # Reset gradients to compute gradients at perturbed point
            optimizer.zero_grad()
            scaler.scale(loss_adv).backward()

            awp._restore()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix(loss=losses.avg)

    return losses.avg


def valid_fn(dataloader, model, criterion, device):
    """
    Validation loop.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    pbar = tqdm(dataloader, total=len(dataloader), desc="Validation")

    with torch.no_grad():
        for data in pbar:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            struct_features = data["struct_features"].to(device)
            labels = data["label"].to(device)

            batch_size = input_ids.size(0)

            y_preds = model(input_ids, attention_mask, struct_features)
            loss = criterion(y_preds.view(-1), labels.view(-1))

            losses.update(loss.item(), batch_size)

            # Apply sigmoid for AUC calculation
            preds.append(y_preds.sigmoid().detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    auc_score = roc_auc_score(ground_truth, predictions)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation AUC: {auc_score}")

    return losses.avg, auc_score


def inference_fn(dataloader, model, device):
    """
    Inference loop for generating predictions.
    """
    model.eval()
    preds = []

    pbar = tqdm(dataloader, total=len(dataloader), desc="Inference")

    with torch.no_grad():
        for data in pbar:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            struct_features = data["struct_features"].to(device)

            y_preds = model(input_ids, attention_mask, struct_features)

            # Apply sigmoid to convert logits to probabilities
            preds.append(y_preds.sigmoid().detach().cpu().numpy())

    predictions = np.concatenate(preds)
    return predictions
