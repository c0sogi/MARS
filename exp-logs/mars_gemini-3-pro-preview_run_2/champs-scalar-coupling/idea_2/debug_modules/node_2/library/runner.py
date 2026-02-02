import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    GroupLogMAE,
    AverageMeter,
)
from library.dataset import MoleculeDataset, collate_dmpnn
from library.model import DMPNN


class Runner:
    def __init__(self, debug=False, load_cached_data=True):
        """
        Initializes the Runner.

        Args:
            debug (bool): If True, runs on a small subset of data.
            load_cached_data (bool): If True, attempts to load pre-processed graph data.
        """
        self.debug = debug
        self.load_cached_data = load_cached_data
        self.device = torch.device(Config.DEVICE)

        # Ensure reproducibility
        set_seed(Config.SEED)

        # Initialize Model
        self.model = DMPNN().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function (MAE is used as a proxy for LogMAE during optimization)
        self.criterion = nn.L1Loss()

        # Placeholders for normalization stats
        self.mean_tensor = None
        self.std_tensor = None

    def _get_normalization_tensors(self, stats):
        """
        Converts the stats dictionary to tensors for vectorized denormalization.
        Returns tensors of shape (NUM_COUPLING_TYPES, 1) on the device.
        """
        means = torch.zeros(Config.NUM_COUPLING_TYPES, device=self.device)
        stds = torch.ones(Config.NUM_COUPLING_TYPES, device=self.device)

        for t_str, t_idx in Config.COUPLING_TYPE_MAP.items():
            if t_str in stats:
                means[t_idx] = stats[t_str]["mean"]
                stds[t_idx] = stats[t_str]["std"]

        return means, stds

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        meter = AverageMeter()

        for batch in train_loader:
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            # Forward pass
            preds = self.model(batch)
            targets = batch["targets"].unsqueeze(-1)

            loss = self.criterion(preds, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            meter.update(loss.item(), batch["targets"].size(0))

        return meter.avg

    def validate(self, val_loader):
        """
        Runs validation and computes the competition metric (LogMAE).
        """
        self.model.eval()
        metric_logger = GroupLogMAE()

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Forward pass (Normalized predictions)
                preds_norm = self.model(batch)
                targets_norm = batch["targets"].unsqueeze(-1)
                type_idxs = batch["type_idxs"]

                # Denormalize
                # Select mean/std based on type_idx
                batch_means = self.mean_tensor[type_idxs].unsqueeze(-1)
                batch_stds = self.std_tensor[type_idxs].unsqueeze(-1)

                preds_denorm = preds_norm * batch_stds + batch_means
                targets_denorm = targets_norm * batch_stds + batch_means

                # Update metric
                metric_logger.update(preds_denorm, targets_denorm, type_idxs)

        avg_log_mae, type_metrics = metric_logger.compute()
        return avg_log_mae, type_metrics

    def train(self):
        """
        Main training loop with Early Stopping and Scheduling.
        """
        print("Initializing Datasets...")
        debug_size = Config.DEBUG_SAMPLE_SIZE if self.debug else None

        # Load Datasets
        train_dataset = MoleculeDataset(
            Config.TRAIN_METADATA_PATH,
            mode="train",
            load_cached=self.load_cached_data,
            debug_size=debug_size,
        )
        val_dataset = MoleculeDataset(
            Config.VAL_METADATA_PATH,
            mode="val",
            load_cached=self.load_cached_data,
            debug_size=debug_size,
        )

        # Prepare Normalization Tensors
        self.mean_tensor, self.std_tensor = self._get_normalization_tensors(
            train_dataset.stats
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_dmpnn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_dmpnn,
            pin_memory=True,
        )

        # Scheduler: Linear Warmup -> Cosine Annealing
        scheduler_warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=Config.WARMUP_EPOCHS,
        )
        scheduler_cosine = CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[Config.WARMUP_EPOCHS],
        )

        # Training Loop
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_score, type_metrics = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.2e} | "
                f"Train MAE: {train_loss:.6f} | Val LogMAE: {val_score:.9f}"
            )

            # Early Stopping & Checkpointing
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_score,
                    Config.MODEL_SAVE_PATH,
                )
                print(f"  -> New best model saved! Score: {best_score:.9f}")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation LogMAE: {best_score:.9f}")

    def predict(self):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print("Starting Inference...")
        debug_size = Config.DEBUG_SAMPLE_SIZE if self.debug else None

        # Load Test Dataset
        test_dataset = MoleculeDataset(
            Config.TEST_METADATA_PATH,
            mode="test",
            load_cached=self.load_cached_data,
            debug_size=debug_size,
        )

        # We need training stats for denormalization
        # If train dataset wasn't loaded in this run, we need to re-compute stats
        # MoleculeDataset always computes stats from TRAIN_METADATA_PATH in _compute_stats
        self.mean_tensor, self.std_tensor = self._get_normalization_tensors(
            test_dataset.stats
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_dmpnn,
            pin_memory=True,
        )

        # Load Best Model
        epoch, score = load_checkpoint(
            Config.MODEL_SAVE_PATH, self.model, device=self.device
        )
        print(f"Loaded model from epoch {epoch} with validation score {score:.9f}")

        self.model.eval()

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Forward
                preds_norm = self.model(batch)
                type_idxs = batch["type_idxs"]

                # Denormalize
                batch_means = self.mean_tensor[type_idxs].unsqueeze(-1)
                batch_stds = self.std_tensor[type_idxs].unsqueeze(-1)
                preds_denorm = preds_norm * batch_stds + batch_means

                # Store
                all_ids.extend(batch["ids"])
                all_preds.extend(preds_denorm.cpu().numpy().flatten())

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
