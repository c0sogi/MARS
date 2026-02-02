import torch
import torch.nn as nn
import timm
import math
import numpy as np
from collections import defaultdict
from library.config import Config
from library.utils import save_checkpoint, calculate_log_loss


def create_model(model_name, num_classes=1, pretrained=True, img_size=224):
    """
    Creates a model using timm.

    Args:
        model_name (str): Name of the model architecture in timm.
        num_classes (int): Number of output classes (1 for binary).
        pretrained (bool): Whether to load pretrained weights.
        img_size (int): Input image size.

    Returns:
        nn.Module: The created model.
    """
    # Create model with timm
    # scriptable=False ensures we can access internal blocks for LLRD if needed

    # Prepare kwargs
    kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "scriptable": False,
    }

    # Only pass img_size to models that require it (e.g., Vision Transformers)
    # Standard CNNs like ResNet and ConvNeXt do not accept img_size in timm
    if "vit" in model_name or "swin" in model_name or "deit" in model_name:
        kwargs["img_size"] = img_size

    model = timm.create_model(model_name, **kwargs)

    return model


def get_optimizer_params(
    model, learning_rate, weight_decay, use_llrd=False, llrd_decay=0.9
):
    """
    Constructs parameter groups for the optimizer, handling Weight Decay exclusion
    and Layer-wise Learning Rate Decay (LLRD) for Transformers.

    Args:
        model (nn.Module): The model to optimize.
        learning_rate (float): Base learning rate.
        weight_decay (float): Weight decay coefficient.
        use_llrd (bool): Whether to apply LLRD (intended for ViT).
        llrd_decay (float): Decay rate for LLRD.

    Returns:
        list: List of parameter groups.
    """

    # Helper to determine if a parameter should have weight decay
    def needs_wd(name, param):
        if param.ndim <= 1:
            return False
        if "bias" in name or "norm" in name or "bn" in name:
            return False
        return True

    if not use_llrd:
        # Standard Parameter Grouping (CNNs)
        # Group 1: Weights (with WD)
        # Group 2: Biases/Norms (no WD)
        wd_params = []
        no_wd_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if needs_wd(name, param):
                wd_params.append(param)
            else:
                no_wd_params.append(param)

        return [
            {"params": wd_params, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": learning_rate, "weight_decay": 0.0},
        ]

    else:
        # Layer-wise Learning Rate Decay (ViT)
        # Assumes timm ViT structure: .blocks, .patch_embed, .head, .norm

        parameter_groups = []

        # 1. Identify layers
        # ViT usually has a 'blocks' Sequential container
        if not hasattr(model, "blocks"):
            # Fallback to standard if structure not found
            print(
                "Warning: LLRD requested but 'blocks' attribute not found. Using standard grouping."
            )
            return get_optimizer_params(
                model, learning_rate, weight_decay, use_llrd=False
            )

        num_layers = len(model.blocks)

        # Scales: Head/Norm = lr, Block N-1 = lr*decay, ..., Embed = lr*decay^(N+1)
        # We group parameters by their "layer depth"

        # Groupings dictionary: depth -> {'wd': [], 'no_wd': []}
        # Depth mapping:
        #   Head/Final Norm -> num_layers + 1
        #   Block i         -> i + 1
        #   Embeddings      -> 0
        groups = defaultdict(lambda: {"wd": [], "no_wd": []})

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine depth
            depth = 0
            if "head" in name or (name.startswith("norm.") and "blocks" not in name):
                depth = num_layers + 1
            elif "blocks" in name:
                # Extract block index from name (e.g., blocks.10.attn...)
                try:
                    parts = name.split(".")
                    block_idx = int(parts[1])
                    depth = block_idx + 1
                except (ValueError, IndexError):
                    depth = 0  # Fallback
            elif "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
                depth = 0
            else:
                depth = 0

            # Assign to WD or No-WD bucket
            if needs_wd(name, param):
                groups[depth]["wd"].append(param)
            else:
                groups[depth]["no_wd"].append(param)

        # Construct optimizer param groups
        for depth in range(num_layers + 2):  # 0 to num_layers + 1
            # Calculate LR for this depth
            # Distance from top: (num_layers + 1) - depth
            # depth = num_layers + 1 (Head) -> dist = 0 -> lr * 1.0
            # depth = 0 (Embed) -> dist = num_layers + 1 -> lr * decay^(N+1)
            dist = (num_layers + 1) - depth
            layer_lr = learning_rate * (llrd_decay**dist)

            # Add WD group
            if groups[depth]["wd"]:
                parameter_groups.append(
                    {
                        "params": groups[depth]["wd"],
                        "lr": layer_lr,
                        "weight_decay": weight_decay,
                    }
                )
            # Add No-WD group
            if groups[depth]["no_wd"]:
                parameter_groups.append(
                    {
                        "params": groups[depth]["no_wd"],
                        "lr": layer_lr,
                        "weight_decay": 0.0,
                    }
                )

        return parameter_groups


class Trainer:
    """
    Handles training, validation, and checkpointing.
    """

    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler, device, config_name
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config_name = config_name
        self.best_loss = float("inf")

        # Loss function: BCEWithLogitsLoss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # [B, 1]

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        count = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Sigmoid for probability
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = running_loss / count

        # Calculate Log Loss metric on full set
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metric_log_loss = calculate_log_loss(all_labels, all_preds)

        return val_loss, metric_log_loss

    def fit(self, epochs):
        print(f"Starting training for {self.config_name}...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, metric_log_loss = self.validate()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val LogLoss: {metric_log_loss:.15f}"
            )

            # Checkpoint Best Model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_path = f"{Config.WORKING_DIR}/{self.config_name}_best.pth"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_loss": self.best_loss,
                    },
                    save_path,
                )

        print(f"Training complete. Best Val Loss: {self.best_loss:.6f}")


def predict(model, loader, device, tta_flip=False):
    """
    Generates predictions for the test set.
    Supports Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Test data loader.
        device (str): Device.
        tta_flip (bool): Whether to use horizontal flip TTA.

    Returns:
        dict: Dictionary mapping id -> probability.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # Forward pass 1: Original
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            if tta_flip:
                # Forward pass 2: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])  # N, C, H, W -> flip W
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(outputs_flipped)

                # Average
                probs = (probs + probs_flipped) / 2.0

            probs = probs.cpu().numpy().flatten()
            ids = ids.numpy().flatten()

            for img_id, prob in zip(ids, probs):
                results[img_id] = prob

    return results
