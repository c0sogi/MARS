import os
import glob
import torch
import torch.nn as nn
import numpy as np
import time
from library.utils import (
    seed_everything,
    calculate_multilabel_auc,
    save_checkpoint,
    save_logs,
)
from library.data import get_dataloaders
from library.models import get_model
from library.optimizer import get_optimizer, get_scheduler


class Trainer:
    """
    Manages the training process, including:
    - Training and Validation loops
    - Metric tracking (Loss, AUC)
    - Snapshot Ensembling (Saving Top-K checkpoints)
    - Early Stopping
    - Learning Rate Scheduling
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config,
        device,
        fold_idx,
        model_name,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.fold_idx = fold_idx
        self.model_name = model_name

        self.criterion = nn.BCEWithLogitsLoss()

        # Checkpoint directory
        self.checkpoint_dir = os.path.join(
            self.config.OUTPUT_DIR, "checkpoints", self.model_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Tracking
        self.logs = []
        self.best_auc_overall = -1.0
        self.patience_counter = 0

        # Snapshot Ensemble: Keep track of top K checkpoints
        # List of dicts: {'path': str, 'score': float, 'epoch': int}
        self.top_k_checkpoints = []

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Apply sigmoid for AUC calculation
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        avg_loss = running_loss / len(self.val_loader)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        auc_score = calculate_multilabel_auc(all_targets, all_preds)

        return avg_loss, auc_score

    def _manage_checkpoints(self, epoch, current_auc, state):
        """
        Manages the Top-K checkpoints logic.
        """
        # Define filename
        filename = f"{self.model_name}_fold_{self.fold_idx}_epoch_{epoch}_auc_{current_auc:.5f}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Add current to list
        new_entry = {"path": filepath, "score": current_auc, "epoch": epoch}

        # We need to decide if we save this checkpoint
        # Logic: If list is not full, save.
        # If list is full, check if current is better than worst in list.

        should_save = False
        if len(self.top_k_checkpoints) < self.config.TOP_K_CHECKPOINTS:
            should_save = True
        else:
            # Get worst score in current top K
            worst_entry = min(self.top_k_checkpoints, key=lambda x: x["score"])
            if current_auc > worst_entry["score"]:
                should_save = True

        if should_save:
            # Save the actual file
            save_checkpoint(state, filepath)
            self.top_k_checkpoints.append(new_entry)

            # Sort descending by score
            self.top_k_checkpoints.sort(key=lambda x: x["score"], reverse=True)

            # Prune if exceeding K
            while len(self.top_k_checkpoints) > self.config.TOP_K_CHECKPOINTS:
                # Remove the last one (worst score)
                to_remove = self.top_k_checkpoints.pop()
                if os.path.exists(to_remove["path"]):
                    os.remove(to_remove["path"])

    def fit(self):
        print(f"Starting training for {self.model_name} - Fold {self.fold_idx}")

        for epoch in range(1, self.config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)
            val_loss, val_auc = self.validate()

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time

            # Logging
            print(
                f"Epoch {epoch}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc:.10f} | "
                f"Time: {elapsed:.2f}s"
            )

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.logs.append(log_entry)

            # Prepare state for saving
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_auc": float(val_auc),
                "config": str(self.config),
            }
            if self.scheduler:
                state["scheduler_state_dict"] = self.scheduler.state_dict()

            # Snapshot Ensemble Logic
            self._manage_checkpoints(epoch, val_auc, state)

            # Early Stopping Logic (based on overall best AUC)
            if val_auc > self.best_auc_overall:
                self.best_auc_overall = val_auc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best AUC: {self.best_auc_overall:.10f}"
                )
                break

        # Save logs
        log_path = os.path.join(self.checkpoint_dir, f"logs_fold_{self.fold_idx}.csv")
        save_logs(self.logs, log_path)

        print(f"Fold {self.fold_idx} finished. Top checkpoints saved.")
        return self.top_k_checkpoints


def run_fold(fold_idx, model_name, config, folds_df, test_df):
    """
    Sets up and runs the training process for a single fold.

    Args:
        fold_idx (int): Index of the current fold.
        model_name (str): Name of the architecture.
        config (Config): Configuration object.
        folds_df (pd.DataFrame): Dataframe with fold information.
        test_df (pd.DataFrame): Dataframe with test information.
    """
    seed_everything(config.SEED + fold_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(fold_idx, folds_df, test_df, config)

    # Model
    model = get_model(model_name, config, device=device)

    # Optimizer & Scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        fold_idx=fold_idx,
        model_name=model_name,
    )

    # Run Training
    trainer.fit()

    # Clear memory
    del model, optimizer, scheduler, trainer, train_loader, val_loader
    torch.cuda.empty_cache()
