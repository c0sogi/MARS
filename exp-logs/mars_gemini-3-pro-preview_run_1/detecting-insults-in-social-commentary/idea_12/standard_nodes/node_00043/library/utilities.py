import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from library.configuration import Config


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Constructs parameter groups for the optimizer with differential learning rates.
    Separates backbone (encoder) and head (decoder) parameters.
    Excludes bias and LayerNorm weights from weight decay.
    """
    # Standard exclusion list for weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []

    # Identify backbone parameters if the model has a 'backbone' attribute
    # This assumes the model wrapper stores the transformer in self.backbone
    if hasattr(model, "backbone"):
        backbone_params = list(model.backbone.named_parameters())
        backbone_ids = {id(p) for n, p in backbone_params}
    else:
        # Fallback: Treat everything as backbone if no distinction exists
        backbone_ids = {id(p) for n, p in model.named_parameters()}

    # Group 1: Backbone parameters with Weight Decay
    optimizer_parameters.append(
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if id(p) in backbone_ids and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": encoder_lr,
        }
    )

    # Group 2: Backbone parameters without Weight Decay
    optimizer_parameters.append(
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if id(p) in backbone_ids and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": encoder_lr,
        }
    )

    # Group 3: Head parameters with Weight Decay
    optimizer_parameters.append(
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if id(p) not in backbone_ids and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": decoder_lr,
        }
    )

    # Group 4: Head parameters without Weight Decay
    optimizer_parameters.append(
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if id(p) not in backbone_ids and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": decoder_lr,
        }
    )

    return optimizer_parameters


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights in the direction of the gradient ascent to flatten the loss landscape.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.2):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Saves the current model weights before perturbation.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def _restore(self):
        """
        Restores the original model weights.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Perturbs the weights (Gradient Ascent on weights).
        Should be called after loss.backward() but before optimizer.step().
        """
        self._save()
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Calculate perturbation: adv_lr * grad / (norm(grad))
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # Perturbation direction
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    # Add perturbation
                    param.data.add_(r_at)
                    # Constraint perturbation to be within epsilon ball
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original weights.
        Should be called after the adversarial forward/backward pass.
        """
        self._restore()
