import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, compute_pearson_score
from library.model import HybridDeberta


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to maximize loss,
    improving model robustness and generalization.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1.0, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, labels, criterion, scaler=None):
        """
        Performs the adversarial attack and backward pass.
        1. Saves current weights.
        2. Perturbs weights based on gradients from the first backward pass.
        3. Computes loss with perturbed weights.
        4. Accumulates gradients.
        5. Restores original weights.
        """
        with torch.no_grad():
            self._save()
            self._attack_step()

        # Forward pass with perturbed weights
        with autocast(enabled=True):
            # Unpack inputs
            input_ids = inputs["input_ids"].to(Config.device)
            attention_mask = inputs["attention_mask"].to(Config.device)
            features = inputs["features"].to(Config.device)

            logits = self.model(input_ids, attention_mask, features)
            adv_loss = criterion(logits, labels)

        # Backward pass with perturbed weights
        if scaler:
            scaler.scale(adv_loss).backward()
        else:
            adv_loss.backward()

        with torch.no_grad():
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


def train_one_epoch(
    model, optimizer, scheduler, dataloader, device, epoch, awp=None, scaler=None
):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

    start = time.time()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        features = data["features"].to(device)
        labels = data["label"].to(device)

        batch_size = input_ids.size(0)

        # 1. First Forward Pass
        with autocast(enabled=True):
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)

        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        # 2. First Backward Pass
        scaler.scale(loss).backward()

        # 3. AWP Attack (if active)
        if awp is not None and epoch >= Config.awp_start_epoch:
            # AWP requires the gradients from the first backward pass to calculate perturbation
            # It then performs a second forward/backward to accumulate robust gradients
            awp.attack_backward(data, labels, criterion, scaler)

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        if (step + 1) % Config.print_freq == 0:
            print(
                f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | Loss: {running_loss / dataset_size:.4f}"
            )

    epoch_loss = running_loss / dataset_size
    print(f"Training Epoch {epoch} Loss: {epoch_loss}")
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

    # Store predictions and targets for Pearson calculation
    all_preds = []
    all_targets = []

    # Mapping indices to score values: 0->0.0, 1->0.25, 2->0.5, 3->0.75, 4->1.0
    score_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            features = data["features"].to(device)
            labels = data["label"].to(device)
            target_scores = data["score"].to(device)  # Float scores for metric

            batch_size = input_ids.size(0)

            with autocast(enabled=True):
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Calculate Expected Value
            probs = torch.softmax(logits, dim=1)  # [Batch, 5]
            expected_scores = torch.sum(probs * score_values, dim=1)  # [Batch]

            all_preds.extend(expected_scores.cpu().numpy())
            all_targets.extend(target_scores.cpu().numpy())

    epoch_loss = running_loss / dataset_size
    pearson_score = compute_pearson_score(np.array(all_preds), np.array(all_targets))

    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation Pearson: {pearson_score}")

    return epoch_loss, pearson_score


def run_fold(fold_idx, train_loader, val_loader):
    """
    Runs the training and validation loop for a single fold.
    """
    print(f"### Starting Fold {fold_idx} ###")

    seed_everything(Config.seed + fold_idx)
    device = Config.device

    # Initialize Model
    model = HybridDeberta()
    model.to(device)

    # Optimizer
    # Separate parameters for weight decay
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    # Scheduler
    num_train_steps = int(
        len(train_loader) * Config.epochs / Config.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # AMP Scaler
    scaler = GradScaler()

    # AWP Initialization
    awp = None
    if Config.use_awp:
        print("Initializing AWP...")
        awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

    best_score = -1.0
    best_model_path = os.path.join(Config.output_dir, f"model_fold_{fold_idx}.bin")

    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch + 1}/{Config.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch + 1, awp, scaler
        )

        # Validate
        val_loss, val_score = valid_one_epoch(model, val_loader, device)

        # Save Best Model
        if val_score > best_score:
            print(f"Score Improved: {best_score} -> {val_score}. Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
        else:
            print(f"Score did not improve from {best_score}.")

    print(f"Fold {fold_idx} Best Pearson Score: {best_score}")

    # Load best model weights before returning (or for inference)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, best_score
