import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from library.utils import AverageMeter, set_seed
from library.model import XLMRobertaForQA


class FGM:
    """
    Fast Gradient Method (FGM) for adversarial training.
    Perturbs input embeddings in the direction of the gradient to maximize loss.
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        """
        Performs the adversarial attack on the embeddings.

        Args:
            epsilon (float): Magnitude of the perturbation.
            emb_name (str): Substring to identify embedding parameters in named_parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # Save original data
                self.backup[name] = param.data.clone()

                # Calculate perturbation
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        """
        Restores the original embedding weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def get_optimizer(model, cfg):
    """
    Configures the AdamW optimizer with Differential Learning Rates.
    Explicitly groups parameters by module to avoid regex fragility.

    Args:
        model (nn.Module): The model to optimize.
        cfg (Config): Configuration object.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    # Explicitly select backbone parameters
    backbone_params = list(model.roberta.parameters())

    # Explicitly select head parameters (Span Head + Relevance Head)
    head_params = list(model.qa_outputs.parameters()) + list(
        model.relevance_head.parameters()
    )

    # Define groups with specific learning rates
    # Note: Weight decay is applied globally as per strategy
    optimizer_grouped_parameters = [
        {
            "params": backbone_params,
            "lr": cfg.lr_backbone,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": head_params,
            "lr": cfg.lr_heads,
            "weight_decay": cfg.weight_decay,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    return optimizer


def train_epoch(model, dataloader, optimizer, scheduler, device, cfg, epoch_idx):
    """
    Trains the model for one epoch using FGM and Loss Averaging.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        cfg: Configuration object.
        epoch_idx: Current epoch index (for logging).
    """
    model.train()

    losses = AverageMeter()
    span_losses = AverageMeter()
    rel_losses = AverageMeter()

    # Initialize FGM if enabled
    fgm = FGM(model) if cfg.use_fgm else None

    # Loss functions
    loss_fct_span = nn.CrossEntropyLoss()
    loss_fct_rel = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        relevance_labels = batch["relevance_labels"].to(device)

        # =====================================================================
        # 1. Clean Pass
        # =====================================================================
        outputs = model(input_ids, attention_mask=attention_mask)

        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        relevance_logits = outputs["relevance_logits"]

        # Calculate Clean Losses
        loss_start = loss_fct_span(start_logits, start_positions)
        loss_end = loss_fct_span(end_logits, end_positions)
        loss_span = (loss_start + loss_end) / 2

        loss_rel = loss_fct_rel(relevance_logits, relevance_labels)

        # Weighted Total Loss
        loss = loss_span + cfg.relevance_loss_weight * loss_rel

        # Backward Clean Loss
        # We scale by 0.5 if using FGM to average clean and adv gradients
        if cfg.use_fgm:
            (loss / 2.0).backward()
        else:
            loss.backward()

        # Update meters
        losses.update(loss.item(), input_ids.size(0))
        span_losses.update(loss_span.item(), input_ids.size(0))
        rel_losses.update(loss_rel.item(), input_ids.size(0))

        # =====================================================================
        # 2. Adversarial Pass (FGM)
        # =====================================================================
        if cfg.use_fgm:
            # Attack: Perturb embeddings based on gradients from clean pass
            fgm.attack(epsilon=cfg.fgm_epsilon)

            # Forward Adv
            outputs_adv = model(input_ids, attention_mask=attention_mask)

            # Calculate Adv Losses
            loss_start_adv = loss_fct_span(outputs_adv["start_logits"], start_positions)
            loss_end_adv = loss_fct_span(outputs_adv["end_logits"], end_positions)
            loss_span_adv = (loss_start_adv + loss_end_adv) / 2

            loss_rel_adv = loss_fct_rel(
                outputs_adv["relevance_logits"], relevance_labels
            )

            loss_adv = loss_span_adv + cfg.relevance_loss_weight * loss_rel_adv

            # Backward Adv Loss (Scaled)
            (loss_adv / 2.0).backward()

            # Restore original embeddings
            fgm.restore()

        # =====================================================================
        # 3. Optimization Step
        # =====================================================================
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Log epoch results
    print(
        f"Epoch {epoch_idx+1}/{cfg.epochs} | "
        f"Loss: {losses.avg:.5f} | "
        f"Span Loss: {span_losses.avg:.5f} | "
        f"Rel Loss: {rel_losses.avg:.5f}"
    )


def train_model(cfg, train_dataset):
    """
    Orchestrates the training process for multiple seeds.
    Trains on the full dataset (merged train+val) for a fixed number of epochs.

    Args:
        cfg (Config): Configuration object.
        train_dataset (Dataset): The training dataset.
    """
    device = cfg.device

    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)

    for seed in cfg.seeds:
        print(f"\n{'='*20} Training Seed {seed} {'='*20}")
        set_seed(seed)

        # Initialize DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Model
        model = XLMRobertaForQA(cfg.model_name)
        model.to(device)

        # Initialize Optimizer
        optimizer = get_optimizer(model, cfg)

        # Initialize Scheduler
        # 10% Warmup
        num_training_steps = len(train_loader) * cfg.epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training Loop
        for epoch in range(cfg.epochs):
            train_epoch(model, train_loader, optimizer, scheduler, device, cfg, epoch)

        # Save Model (Converged State)
        save_path = os.path.join(cfg.output_dir, f"model_seed_{seed}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Clear memory for next seed
        del model, optimizer, scheduler, train_loader
        torch.cuda.empty_cache()
