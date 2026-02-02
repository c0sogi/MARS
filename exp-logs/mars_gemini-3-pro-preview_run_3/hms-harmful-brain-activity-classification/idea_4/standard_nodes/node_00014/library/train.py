import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_logger, kl_divergence_score
from library.data_loader import get_dataloaders
from library.models import HybridEEGModel


class Trainer:
    def __init__(
        self,
        model,
        device,
        optimizer,
        scheduler,
        criterion,
        patience=Config.PATIENCE,
        model_path=Config.MODEL_PATH,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.patience = patience
        self.model_path = model_path
        self.logger = get_logger("trainer")
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        start_time = time.time()

        for batch_idx, (raw_x, spec_x, targets) in enumerate(dataloader):
            raw_x = raw_x.to(self.device)
            spec_x = spec_x.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # Model outputs Softmax probabilities
            outputs = self.model(raw_x, spec_x)

            # KLDivLoss expects input as Log-Probabilities and target as Probabilities
            # We apply log to the softmax outputs
            loss = self.criterion(torch.log(outputs + 1e-15), targets)

            loss.backward()
            self.optimizer.step()

            batch_size = raw_x.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        elapsed = time.time() - start_time

        self.logger.info(
            f"Epoch {epoch_idx} - Train Loss: {epoch_loss} - Time: {elapsed:.2f}s"
        )
        return epoch_loss

    def validate_one_epoch(self, dataloader, epoch_idx):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        # Lists to store all preds and targets for metric calculation if needed globally,
        # but average batch loss is sufficient for KLDiv optimization.

        with torch.no_grad():
            for raw_x, spec_x, targets in dataloader:
                raw_x = raw_x.to(self.device)
                spec_x = spec_x.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(raw_x, spec_x)

                # Calculate loss using the same criterion
                loss = self.criterion(torch.log(outputs + 1e-15), targets)

                batch_size = raw_x.size(0)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        # We print full precision as requested
        self.logger.info(f"Epoch {epoch_idx} - Val Loss: {epoch_loss}")

        return epoch_loss

    def fit(self, train_loader, val_loader, epochs):
        self.logger.info(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.validate_one_epoch(val_loader, epoch)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.logger.info(
                    f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                )
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                self.early_stop_counter += 1
                self.logger.info(
                    f"Validation loss did not improve. Counter: {self.early_stop_counter}/{self.patience}"
                )

            if self.early_stop_counter >= self.patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val Loss: {self.best_val_loss}")


def train_model(
    debug=Config.DEBUG,
    load_cached=True,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    num_workers=Config.NUM_WORKERS,
):
    """
    Main function to set up and run the training process.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Data Loaders
    logger = get_logger("train_setup")
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        debug=debug,
        load_cached=load_cached,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 4. Model
    logger.info("Initializing HybridEEGModel...")
    model = HybridEEGModel(num_classes=Config.N_CLASSES, pretrained_spec=True)
    model = model.to(device)

    # 5. Optimizer & Loss
    # KLDivLoss with reduction='batchmean' aligns with the mathematical definition of KL divergence
    criterion = nn.KLDivLoss(reduction="batchmean")

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        patience=Config.PATIENCE,
        model_path=Config.MODEL_PATH,
    )

    # 7. Start Training
    trainer.fit(train_loader, val_loader, epochs)

    return trainer.best_val_loss
