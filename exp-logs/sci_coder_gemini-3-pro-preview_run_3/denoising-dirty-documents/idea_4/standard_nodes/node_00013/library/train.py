import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import DnCNN
from library.data import prepare_data, DenoisingDataset


class Trainer:
    """
    Manages the training process for the DnCNN model.
    """

    def __init__(self):
        """
        Initializes the model, optimizer, loss function, and scheduler.
        """
        self.device = Config.DEVICE

        # Initialize Model
        self.model = DnCNN(
            depth=Config.DEPTH,
            n_channels=Config.N_CHANNELS,
            image_channels=Config.IN_CHANNELS,
        ).to(self.device)

        # Loss Function: MSE between predicted noise and actual noise
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

    def train(self, load_cached_data=True):
        """
        Executes the training loop with validation and early stopping.

        Args:
            load_cached_data (bool): Whether to load pre-processed patches from disk.
        """
        # --- Data Preparation ---
        # Load training data
        train_data = prepare_data(
            Config.TRAIN_METADATA_PATH,
            Config.TRAIN_CACHE_PATH,
            load_cached_data=load_cached_data,
        )

        # Load validation data
        val_data = prepare_data(
            Config.VAL_METADATA_PATH,
            Config.VAL_CACHE_PATH,
            load_cached_data=load_cached_data,
        )

        # Create Datasets
        train_dataset = DenoisingDataset(train_data, augment=Config.AUGMENT_DATA)
        val_dataset = DenoisingDataset(val_data, augment=False)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting training on {self.device}...")

        best_val_loss = float("inf")
        patience_counter = 0

        # --- Training Loop ---
        for epoch in range(Config.EPOCHS):
            self.model.train()
            running_loss = 0.0

            for noisy, clean in train_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                # The model learns to predict the noise residual: R(x) = Noisy - Clean
                target_noise = noisy - clean

                self.optimizer.zero_grad()

                # Forward pass
                pred_noise = self.model(noisy)

                # Compute loss
                loss = self.criterion(pred_noise, target_noise)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * noisy.size(0)

            epoch_loss = running_loss / len(train_dataset)

            # --- Validation Loop ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy = noisy.to(self.device)
                    clean = clean.to(self.device)

                    target_noise = noisy - clean
                    pred_noise = self.model(noisy)

                    loss = self.criterion(pred_noise, target_noise)
                    val_loss += loss.item() * noisy.size(0)

            val_loss /= len(val_dataset)

            # Print metrics with full precision
            print(f"Epoch {epoch+1}: Train Loss {epoch_loss}, Val Loss {val_loss}")

            # Update Scheduler
            self.scheduler.step()

            # --- Early Stopping & Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break
