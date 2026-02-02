import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import MaterialsDataset, collate_fn
from library.model import GDCC_WDS


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, evaluation, and saving of the model.
    """

    def __init__(self, model, device, learning_rate, weight_decay, patience):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # Cite debug_lesson_2: Remove Deprecated verbose Argument from PyTorch Scheduler Constructors
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )
        # Targets are already log(1+x) transformed in the dataset.
        # MSE on log-transformed targets is equivalent to MSLE on original targets.
        self.criterion = nn.MSELoss()
        self.patience = patience
        self.best_val_loss = float("inf")
        self.counter = 0

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Move data to device
            atomic_features = batch["atomic_features"].to(self.device)
            global_features = batch["global_features"].to(self.device)
            batch_indices = batch["batch_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            # Note: num_graphs is the batch size (len(ids))
            num_graphs = len(batch["ids"])
            outputs = self.model(
                atomic_features, global_features, batch_indices, num_graphs
            )

            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * num_graphs

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        # To compute column-wise metrics
        total_sq_error = torch.zeros(Config.OUTPUT_DIM).to(self.device)
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                atomic_features = batch["atomic_features"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                batch_indices = batch["batch_indices"].to(self.device)
                targets = batch["targets"].to(self.device)

                num_graphs = len(batch["ids"])
                outputs = self.model(
                    atomic_features, global_features, batch_indices, num_graphs
                )

                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * num_graphs

                # Column-wise squared error
                sq_error = (outputs - targets) ** 2
                total_sq_error += sq_error.sum(dim=0)
                total_samples += num_graphs

        epoch_loss = running_loss / total_samples

        # Calculate RMSLE for each column (targets are already log transformed)
        # RMSLE = sqrt(mean((log(1+y_pred) - log(1+y_true))^2))
        # Here outputs ~ log(1+y_pred), targets ~ log(1+y_true)
        column_rmsle = torch.sqrt(total_sq_error / total_samples)

        return epoch_loss, column_rmsle.cpu().numpy()

    def fit(self, train_loader, val_loader, num_epochs, save_path):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_col_rmsle = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # RMSLE is sqrt of MSE on log-transformed data
            train_rmsle = np.sqrt(train_loss)
            val_rmsle = np.sqrt(val_loss)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s")
            print(
                f"  Train Loss (MSE): {train_loss:.8f} | Train RMSLE: {train_rmsle:.8f}"
            )
            print(f"  Val Loss (MSE):   {val_loss:.8f} | Val RMSLE:   {val_rmsle:.8f}")
            print(
                f"  Val Column RMSLE: Formation E={val_col_rmsle[0]:.8f}, Bandgap={val_col_rmsle[1]:.8f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> New best model saved to {save_path}")
            else:
                self.counter += 1
                print(f"  -> Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {self.best_val_loss:.8f}")


def train_model(
    max_samples=None,
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Main function to setup and run the training process.

    Args:
        max_samples (int, optional): Limit dataset size for debugging.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Set seed
    set_seed(Config.SEED)

    # 2. Ensure directories exist
    Config.make_dirs()

    # 3. Create Datasets
    print("Initializing Datasets...")
    train_dataset = MaterialsDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        mode="train",
        max_samples=max_samples,
        load_cached_data=load_cached_data,
    )

    val_dataset = MaterialsDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        mode="val",
        max_samples=max_samples,
        load_cached_data=load_cached_data,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 5. Initialize Model
    print("Initializing Model...")
    model = GDCC_WDS()

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        device=Config.DEVICE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 7. Start Training
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        save_path=Config.BEST_MODEL_PATH,
    )


if __name__ == "__main__":
    # Example usage (can be adjusted or called from another script)
    # Using a smaller number of epochs for demonstration if run directly
    # In a real scenario, use Config.NUM_EPOCHS
    train_model()
