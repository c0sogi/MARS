import os
import time
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import (
    seed_everything,
    mixup_data,
    mixup_criterion,
    calculate_overall_lwlrap,
)
from library.dataset import get_dataloaders, get_class_names
from library.model import AudioEfficientNet


class Trainer:
    def __init__(self):
        """
        Initialize the Trainer with configuration, model, optimizer, and loss function.
        """
        # Set reproducible seed
        seed_everything(Config.SEED)

        self.device = Config.DEVICE
        self.num_classes = Config.NUM_CLASSES

        # Initialize Model
        print(f"Initializing model: {Config.MODEL_NAME}")
        self.model = AudioEfficientNet()
        self.model.to(self.device)

        # Loss Function
        # BCEWithLogitsLoss is more stable than Sigmoid + BCELoss
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler placeholder (initialized in fit)
        self.scheduler = None

        # Best model state
        self.best_model_state = None
        self.best_score = -np.inf

    def train_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training with Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        start_time = time.time()

        for batch_idx, (data, target, _) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            batch_size = data.size(0)

            self.optimizer.zero_grad()

            # Apply Mixup
            if Config.USE_MIXUP:
                data, target_a, target_b, lam = mixup_data(
                    data, target, Config.MIXUP_ALPHA
                )
                output = self.model(data)
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch_idx+1}/{Config.EPOCHS} | Train Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s"
        )
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and LWLRAP score.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target, _ in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                batch_size = data.size(0)

                # Forward pass
                logits = self.model(data)
                loss = self.criterion(logits, target)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(logits)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                all_preds.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Calculate Metric
        lwlrap = calculate_overall_lwlrap(all_targets, all_preds)

        return epoch_loss, lwlrap

    def fit(self, load_cached_data=True):
        """
        Main training loop with Early Stopping.
        """
        print("Loading data...")
        train_loader, val_loader, test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # Initialize Scheduler
        steps_per_epoch = len(train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=Config.MAX_LR,
            steps_per_epoch=steps_per_epoch,
            epochs=Config.EPOCHS,
            pct_start=0.1,  # 10% of training for warmup
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        print(f"Starting training for {Config.EPOCHS} epochs...")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Train
            self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_score = self.validate(val_loader)
            print(
                f"Epoch {epoch+1} Validation | Loss: {val_loss} | LWLRAP: {val_score}"
            )

            # Early Stopping Check
            if val_score > self.best_score:
                print(
                    f"Validation score improved ({self.best_score} --> {val_score}). Saving model..."
                )
                self.best_score = val_score
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save checkpoint to disk
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.best_model_state, save_path)
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation LWLRAP: {self.best_score}")

        # Load best weights for prediction
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return test_loader

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Generating predictions on test set...")
        self.model.eval()

        predictions = []
        fnames = []

        with torch.no_grad():
            for data, _, fname_batch in test_loader:
                data = data.to(self.device)

                # Forward pass
                logits = self.model(data)
                probs = torch.sigmoid(logits)

                predictions.append(probs.cpu().numpy())
                fnames.extend(fname_batch)

        # Combine predictions
        predictions = np.vstack(predictions)

        # Get class names for columns
        class_names = get_class_names()

        # Create DataFrame
        sub_df = pd.DataFrame(predictions, columns=class_names)
        sub_df.insert(0, "fname", fnames)

        # Save submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")


def run_training():
    """
    Helper function to instantiate the trainer and run the pipeline.
    """
    trainer = Trainer()

    # Train and Validate
    test_loader = trainer.fit(load_cached_data=True)

    # Predict and Submit
    trainer.predict(test_loader)
