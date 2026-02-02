import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from library.config import Config
from library.utils import seed_everything, MCRMSE, create_submission_file
from library.data import get_dataloaders
from library.model import DeepHierarchicalBiGRU


class Trainer:
    """
    Manages the training, validation, and checkpointing process.
    """

    def __init__(self, model, optimizer, scheduler, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.criterion = nn.MSELoss()
        self.best_score = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training with Deep Supervision.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, adjacency, targets) in enumerate(train_loader):
            features = features.to(self.device)
            adjacency = adjacency.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: returns list of outputs [head1, head2, head3, head4]
            outputs = self.model(features, adjacency)

            # Calculate Deep Supervision Loss
            # Final head loss
            loss_final = self.criterion(outputs[-1], targets)

            # Auxiliary heads loss
            loss_aux = 0.0
            for i in range(len(outputs) - 1):
                loss_aux += self.criterion(outputs[i], targets)

            # Total loss
            total_loss = loss_final + self.config.DEEP_SUPERVISION_WEIGHT * loss_aux

            # Backward pass
            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            running_loss += total_loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using the competition metric.
        Only the final head is used for validation.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, adjacency, targets in val_loader:
                features = features.to(self.device)
                adjacency = adjacency.to(self.device)

                # Forward pass
                outputs = self.model(features, adjacency)

                # Use only the final head for prediction
                final_pred = outputs[-1].cpu()

                all_preds.append(final_pred)
                all_targets.append(targets)

        # Concatenate all batches
        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        # Calculate MCRMSE using the utility function (handles slicing internally)
        score = MCRMSE(y_true, y_pred)
        return score

    def fit(self, train_loader, val_loader, num_epochs):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}...")
        patience_counter = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_score} | "
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping and Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                # print(f"  New best model saved! Score: {self.best_score:.6f}")
            else:
                patience_counter += 1
                # print(f"  No improvement. Patience: {patience_counter}/{self.config.PATIENCE}")

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation MCRMSE: {self.best_score}")


def run_training_pipeline():
    """
    Main function to execute the training and inference pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = DeepHierarchicalBiGRU(
        input_channels=Config.INPUT_CHANNELS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        num_targets=Config.NUM_TARGETS,
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # 5. Training
    trainer = Trainer(model, optimizer, scheduler, device, Config)
    trainer.fit(train_loader, val_loader, Config.NUM_EPOCHS)

    # 6. Inference on Test Set
    print("Running inference on test set...")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for features, adjacency, _ in test_loader:
            features = features.to(device)
            adjacency = adjacency.to(device)

            outputs = model(features, adjacency)

            # Use final head
            final_pred = outputs[-1].cpu().numpy()
            all_preds.append(final_pred)

    # Concatenate predictions: (N_test, 107, 5)
    test_preds = np.concatenate(all_preds, axis=0)

    # 7. Create Submission
    print("Generating submission file...")
    create_submission_file(test_ids, test_preds, Config.SUBMISSION_PATH)
    print("Done.")
