import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from library.config import Config
from library.utils import AverageMeter, get_logger

logger = get_logger("Trainer")


class Trainer:
    """
    Trainer class to handle the training loop for a single model seed.
    Encapsulates optimization, loss calculation, and model saving.
    """

    def __init__(self, model, train_loader, device, seed):
        """
        Args:
            model: The MultiTaskXLMR model instance.
            train_loader: DataLoader for the training set.
            device: 'cuda' or 'cpu'.
            seed: The random seed associated with this training run.
        """
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.seed = seed
        self.optimizer = self.get_optimizer_params()

    def get_optimizer_params(self):
        """
        Defines parameter groups with differential learning rates and global weight decay.
        Uses explicit module references to separate the backbone from the heads.
        """
        # Explicit parameter grouping as requested to apply specific LRs
        # Weight decay is applied to ALL parameters (including bias/LayerNorm) per instructions
        optimizer_grouped_parameters = [
            {
                "params": self.model.backbone.parameters(),
                "lr": Config.LR_BACKBONE,
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": self.model.span_head.parameters(),
                "lr": Config.LR_HEAD,
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": self.model.relevance_head.parameters(),
                "lr": Config.LR_HEAD,
                "weight_decay": Config.WEIGHT_DECAY,
            },
        ]

        return AdamW(optimizer_grouped_parameters)

    def train_one_epoch(self, epoch_idx):
        """
        Trains the model for one epoch.
        Computes the combined loss: Span Loss + (0.5 * Relevance Loss).

        Args:
            epoch_idx (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()

        losses = AverageMeter()
        span_losses = AverageMeter()
        rel_losses = AverageMeter()

        loss_fct_span = nn.CrossEntropyLoss()
        loss_fct_rel = nn.BCEWithLogitsLoss()

        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)
            relevance_labels = batch["relevance_labels"].to(self.device)

            # Forward pass
            start_logits, end_logits, rel_logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # --- Loss Calculation ---
            # 1. Span Loss (Start + End)
            start_loss = loss_fct_span(start_logits, start_positions)
            end_loss = loss_fct_span(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2

            # 2. Relevance Loss
            # Flatten logits and labels for BCE
            rel_loss = loss_fct_rel(rel_logits.view(-1), relevance_labels.view(-1))

            # 3. Total Loss
            total_loss = span_loss + (Config.RELEVANCE_LOSS_WEIGHT * rel_loss)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update metrics
            batch_size = input_ids.size(0)
            losses.update(total_loss.item(), batch_size)
            span_losses.update(span_loss.item(), batch_size)
            rel_losses.update(rel_loss.item(), batch_size)

            # Log periodically
            if step % 50 == 0:
                logger.info(
                    f"Seed {self.seed} | Epoch {epoch_idx} | Step {step} | "
                    f"Loss: {losses.avg:.5f} | Span: {span_losses.avg:.5f} | Rel: {rel_losses.avg:.5f}"
                )

        return losses.avg

    def train(self):
        """
        Runs the full training loop for the number of epochs specified in Config.
        Saves the final model state to the working directory.
        """
        logger.info(f"Starting training for Seed {self.seed} on device {self.device}")
        self.model.to(self.device)

        for epoch in range(1, Config.EPOCHS + 1):
            avg_loss = self.train_one_epoch(epoch)
            logger.info(
                f"Seed {self.seed} | Epoch {epoch} | Average Loss: {avg_loss:.6f}"
            )

        # Save the model
        # Ensure working directory exists (handled in Config, but safe to check)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        save_name = f"model_seed_{self.seed}.pth"
        save_path = os.path.join(Config.WORKING_DIR, save_name)

        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model for Seed {self.seed} saved to {save_path}")

        return save_path
