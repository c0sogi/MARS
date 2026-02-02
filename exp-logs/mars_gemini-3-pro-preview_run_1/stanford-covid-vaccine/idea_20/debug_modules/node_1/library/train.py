import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    generate_submission_file,
)
from library.data import get_dataloaders
from library.model import Net
from library.loss import MaskedMSELoss


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        logger,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.best_metric = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            sequence = batch["sequence"].to(self.device)
            loop_type = batch["loop_type"].to(self.device)
            distance = batch["distance"].to(self.device)
            targets = batch["target"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            outputs = self.model(sequence, loop_type, distance)

            # Compute loss
            loss = self.criterion(outputs, targets, mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.max_grad_norm
            )

            self.optimizer.step()

            loss_meter.update(loss.item(), sequence.size(0))

        return loss_meter.avg

    def validate(self):
        self.model.eval()

        # Accumulators for global MCRMSE calculation
        # We need to sum squared errors per column and count valid pixels per column
        total_squared_error = torch.zeros(Config.n_targets, device=self.device)
        total_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                sequence = batch["sequence"].to(self.device)
                loop_type = batch["loop_type"].to(self.device)
                distance = batch["distance"].to(self.device)
                targets = batch["target"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(sequence, loop_type, distance)

                # Select valid positions
                valid_outputs = outputs[mask]
                valid_targets = targets[mask]

                # Accumulate squared errors: sum((pred - true)^2, dim=0)
                # valid_outputs shape: (N_valid_pixels, n_targets)
                squared_diff = (valid_outputs - valid_targets) ** 2
                total_squared_error += squared_diff.sum(dim=0)
                total_count += valid_outputs.size(0)

        # Compute RMSE per column
        # MSE = Sum_Squared_Error / Count
        mse_per_col = total_squared_error / total_count
        rmse_per_col = torch.sqrt(mse_per_col)

        # MCRMSE is the mean of the column RMSEs
        mcrmse = torch.mean(rmse_per_col).item()

        return mcrmse

    def fit(self, epochs, patience):
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_mcrmse = self.validate()

            # Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_mcrmse)

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_mcrmse}"  # Full precision as requested
            )

            # Checkpointing
            is_best = val_mcrmse < self.best_metric
            if is_best:
                self.best_metric = val_mcrmse
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "best_metric": self.best_metric,
                    },
                    is_best=True,
                    checkpoint_dir=Config.working_dir,
                )
                self.logger.info(
                    f"New best model saved with MCRMSE: {self.best_metric}"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

        self.logger.info(f"Training complete. Best MCRMSE: {self.best_metric}")


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            distance = batch["distance"].to(device)

            # Forward pass
            # Output shape: (B, L, 3)
            outputs = model(sequence, loop_type, distance)

            # Move to CPU and collect
            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches: (N_samples, L, 3)
    return np.concatenate(all_preds, axis=0)


def run_training():
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger("Train", log_file=os.path.join(Config.working_dir, "train.log"))

    # 2. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    logger.info("Initializing model...")
    model = Net().to(Config.device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.scheduler_mode,
        factor=Config.scheduler_factor,
        patience=Config.scheduler_patience,
        min_lr=Config.min_lr,
        verbose=True,
    )

    criterion = MaskedMSELoss()

    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=Config.device,
        logger=logger,
    )

    trainer.fit(epochs=Config.epochs, patience=Config.es_patience)

    # 6. Inference
    logger.info("Loading best model for inference...")
    best_model_path = Config.model_save_path
    load_checkpoint(best_model_path, model, device=Config.device)

    logger.info("Generating predictions on test set...")
    predictions = predict(model, test_loader, Config.device)

    # 7. Submission Generation
    logger.info("Preparing submission file...")
    # Load test metadata to get IDs and Sequences
    df_test = pd.read_parquet(Config.test_file)
    ids = df_test["id"].tolist()
    sequences = df_test["sequence"].tolist()

    generate_submission_file(
        ids=ids,
        sequences=sequences,
        predictions=predictions,
        output_path=Config.submission_path,
    )

    logger.info("Process finished successfully.")
