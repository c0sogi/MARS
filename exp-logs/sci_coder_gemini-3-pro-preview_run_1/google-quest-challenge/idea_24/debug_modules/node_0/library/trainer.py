import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import compute_spearmanr_metric, setup_logger


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE
        self.logger = setup_logger(
            "Trainer", os.path.join(Config.WORKING_DIR, "train.log")
        )

        # Move model to device
        self.model.to(self.device)

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer with Differential Learning Rates
        self.optimizer = self._get_optimizer()

        # Scheduler with Phantom Scheduling Strategy
        # We calculate steps based on SCHEDULER_TOTAL_EPOCHS (7) to get a gentle decay,
        # but we will explicitly stop the training loop after EPOCHS (3).
        num_update_steps_per_epoch = len(self.train_loader) // Config.ACCUMULATION_STEPS
        max_train_steps = num_update_steps_per_epoch * Config.SCHEDULER_TOTAL_EPOCHS
        warmup_steps = int(max_train_steps * Config.WARMUP_RATIO)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        self.best_score = -float("inf")

    def _get_optimizer(self):
        """
        Sets up AdamW optimizer with differential learning rates.
        Separates parameters into backbone (low LR) and head/others (high LR).
        """
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer_grouped_parameters = [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ]

        return torch.optim.AdamW(optimizer_grouped_parameters)

    def train_one_epoch(self, epoch_idx):
        """
        Trains the model for one epoch.
        Handles Head Warmup (freezing backbone) and Gradient Accumulation.
        """
        self.model.train()

        # Head Warmup Logic
        if epoch_idx < Config.FREEZE_BACKBONE_EPOCHS:
            self.logger.info(f"Epoch {epoch_idx + 1}: Freezing backbone (Head Warmup)")
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            self.logger.info(f"Epoch {epoch_idx + 1}: Unfreezing backbone")
            for param in self.model.backbone.parameters():
                param.requires_grad = True

        running_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            labels = batch["labels"]

            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS
            loss.backward()

            running_loss += loss.item() * Config.ACCUMULATION_STEPS

            # Step optimizer and scheduler
            # Perform step if accumulation count reached OR it's the last batch
            if (step + 1) % Config.ACCUMULATION_STEPS == 0 or (step + 1) == len(
                self.train_loader
            ):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        avg_loss = running_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch_idx + 1} Train Loss: {avg_loss}")

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns the mean column-wise Spearman's correlation.
        """
        self.model.eval()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                labels = batch["labels"]
                logits = self.model(batch)
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu())
                targets_list.append(labels.cpu())

        preds = torch.cat(preds_list, dim=0).numpy()
        targets = torch.cat(targets_list, dim=0).numpy()

        score = compute_spearmanr_metric(preds, targets)
        return score

    def fit(self):
        """
        Main training loop. Runs for Config.EPOCHS.
        Saves the best model based on validation score.
        """
        self.logger.info("Starting training...")

        for epoch in range(Config.EPOCHS):
            self.train_one_epoch(epoch)
            val_score = self.validate()

            # Print full precision as requested
            self.logger.info(f"Epoch {epoch + 1} Validation Score: {val_score}")

            if val_score > self.best_score:
                self.best_score = val_score
                self.save_model("best_model.pth")
                self.logger.info(f"New best model saved with score: {val_score}")

        self.logger.info(f"Training complete. Best Score: {self.best_score}")

    def save_model(self, filename):
        """
        Saves the model state dict to the working directory.
        """
        path = os.path.join(Config.WORKING_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): Test data loader.

        Returns:
            tuple: (predictions numpy array, qa_ids numpy array)
        """
        self.model.eval()
        preds_list = []
        qa_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                logits = self.model(batch)
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                qa_ids_list.append(batch["qa_id"].cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        qa_ids = np.concatenate(qa_ids_list, axis=0)
        return preds, qa_ids


def create_submission(trainer, test_loader, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions using the trainer and saves them to a CSV file.

    Args:
        trainer (Trainer): The trained trainer instance.
        test_loader (DataLoader): DataLoader for the test set.
        output_path (str): Path to save the submission CSV.
    """
    trainer.logger.info("Generating submission...")
    preds, qa_ids = trainer.predict(test_loader)

    # Create DataFrame
    df = pd.DataFrame(preds, columns=Config.TARGET_COLS)
    df.insert(0, "qa_id", qa_ids)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    trainer.logger.info(f"Submission saved to {output_path}")
