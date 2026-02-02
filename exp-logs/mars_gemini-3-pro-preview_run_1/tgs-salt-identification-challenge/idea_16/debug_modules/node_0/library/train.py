import os
import time
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, AverageMeter, save_numpy_cache
from library.dataset import get_loaders
from library.model import DeepResUNet
from library.losses import CurriculumLoss
from library.metrics import calculate_map_at_thresholds


class Trainer:
    """
    Handles the training, validation, and checkpointing logic for the Salt Segmentation task.
    """

    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler, criterion, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        # Metrics tracking
        self.best_map_global = 0.0
        self.best_map_cycle_2 = 0.0
        self.best_map_cycle_3 = 0.0

        # Cycle definitions based on Config
        self.cycle_len = Config.EPOCHS_PER_CYCLE

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter("Loss")

        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            logits = self.model(images)

            # Calculate loss (CurriculumLoss handles phase switching based on epoch)
            loss = self.criterion(logits, masks, epoch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()
        map_scores = []

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)

                # Apply sigmoid to get probabilities for metric calculation
                probs = torch.sigmoid(logits)

                # Calculate mAP (competition metric)
                # Using 0.5 as pixel probability threshold
                batch_map = calculate_map_at_thresholds(probs, masks, threshold=0.5)
                map_scores.append(batch_map)

        return np.mean(map_scores)

    def fit(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_map = self.validate()

            # Update Scheduler (CosineAnnealingWarmRestarts steps per epoch)
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Determine current cycle (1-based index for logging/saving logic)
            # Cycle 1: Epochs 0-49
            # Cycle 2: Epochs 50-99
            # Cycle 3: Epochs 100-149
            current_cycle = (epoch // self.cycle_len) + 1

            print(
                f"Epoch {epoch+1}/{num_epochs} [Cycle {current_cycle}] - "
                f"Time: {elapsed:.1f}s - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val mAP: {val_map:.10f}"
            )

            # Checkpointing Logic
            self._save_checkpoints(epoch, current_cycle, val_map)

    def _save_checkpoints(self, epoch, current_cycle, val_map):
        # Save Global Best
        if val_map > self.best_map_global:
            self.best_map_global = val_map
            torch.save(
                self.model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

        # Save Cycle 2 Best (Epochs 50-99)
        if current_cycle == 2:
            if val_map > self.best_map_cycle_2:
                self.best_map_cycle_2 = val_map
                print(f"  -> New Best Cycle 2 Model (mAP: {val_map:.6f})")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.CHECKPOINT_DIR, "best_cycle_2.pth"),
                )

        # Save Cycle 3 Best (Epochs 100-149)
        if current_cycle == 3:
            if val_map > self.best_map_cycle_3:
                self.best_map_cycle_3 = val_map
                print(f"  -> New Best Cycle 3 Model (mAP: {val_map:.6f})")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.CHECKPOINT_DIR, "best_cycle_3.pth"),
                )


def train_model():
    """
    Main execution function to setup and run the training pipeline.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, _ = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = DeepResUNet()
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = CurriculumLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Warm Restarts
    # T_0 is the number of epochs for the first restart.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.EPOCHS_PER_CYCLE, T_mult=1, eta_min=1e-6
    )

    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
    )

    trainer.fit(num_epochs=Config.TOTAL_EPOCHS)

    print("Training complete.")
    print(f"Best Global mAP: {trainer.best_map_global:.6f}")
    print(f"Best Cycle 2 mAP: {trainer.best_map_cycle_2:.6f}")
    print(f"Best Cycle 3 mAP: {trainer.best_map_cycle_3:.6f}")
