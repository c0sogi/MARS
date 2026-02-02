import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, WeightedL1Loss
from library.dataset import get_dataloaders
from library.model import DP_GI_BiLSTM, train_one_epoch, validate


class Trainer:
    """
    Manages the training procedure for the Ventilator Pressure Prediction task.
    Implements the Stretched-Horizon Convergence Protocol.
    """

    def __init__(self):
        # 1. Reproducibility & Hardware
        seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)
        print(f"Trainer initialized on device: {self.device}")

        # 2. Data Loading
        # get_dataloaders handles caching internally as per library.dataset
        self.train_loader, self.val_loader, _ = get_dataloaders(load_cached_data=True)

        # 3. Model Initialization
        self.model = DP_GI_BiLSTM(input_dim=Config.INPUT_DIM).to(self.device)

        # 4. Optimization
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Stretched-Horizon Scheduler
        # T_max matches the total number of epochs to keep LR higher for longer
        self.epochs = Config.get_epochs()
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=Config.ETA_MIN
        )

        # 5. Loss Function
        # Weighted L1 Loss (Inspiratory vs Expiratory)
        self.criterion = WeightedL1Loss(
            inspiratory_weight=Config.LOSS_INSPIRATORY_WEIGHT,
            expiratory_weight=Config.LOSS_EXPIRATORY_WEIGHT,
        )

        # 6. Tracking
        self.best_mae = float("inf")

    def fit(self):
        """
        Executes the training loop.
        """
        print(f"Starting training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            start_time = time.time()

            # Train Step
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )

            # Validation Step
            val_loss, val_mae = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision for Validation MAE)
            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_mae:.16f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpoint Strategy: Save Best Model
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"  >>> New Best Model Saved! MAE: {self.best_mae:.16f}")

        print(f"Training complete. Best Validation MAE: {self.best_mae:.16f}")
