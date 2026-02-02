import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_accuracy, AverageMeter
from library.data_processing import get_dataloaders
from library.model import DCNV2


class Trainer:
    """
    Manages the training, validation, and prediction processes for the DCN-V2 model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, test_ids, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_ids = test_ids
        self.device = device

        # Define Loss, Optimizer, and Scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()

        for batch_idx, (x_cont, x_cat, target) in enumerate(self.train_loader):
            x_cont = x_cont.to(self.device)
            x_cat = x_cat.to(self.device)
            target = target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(x_cont, x_cat)
            loss = self.criterion(outputs, target)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            acc = calculate_accuracy(outputs, target)
            losses.update(loss.item(), x_cont.size(0))
            accuracies.update(acc, x_cont.size(0))

        return losses.avg, accuracies.avg

    def validate(self):
        """Runs evaluation on the validation set."""
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for x_cont, x_cat, target in self.val_loader:
                x_cont = x_cont.to(self.device)
                x_cat = x_cat.to(self.device)
                target = target.to(self.device)

                outputs = self.model(x_cont, x_cat)
                loss = self.criterion(outputs, target)
                acc = calculate_accuracy(outputs, target)

                losses.update(loss.item(), x_cont.size(0))
                accuracies.update(acc, x_cont.size(0))

        return losses.avg, accuracies.avg

    def fit(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update scheduler based on validation accuracy
            self.scheduler.step(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}")
            print(
                f"Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc}"
            )  # Full precision printing

            # Early Stopping Logic
            if val_acc > best_acc:
                best_acc = val_acc
                # Crucial: Deep copy to prevent mutation
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                # Save checkpoint immediately
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        print(f"Training complete. Best Validation Accuracy: {best_acc}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

    def predict(self):
        """
        Generates predictions for the test set and saves submission file.
        """
        print("Generating predictions on test set...")
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for x_cont, x_cat in self.test_loader:
                x_cont = x_cont.to(self.device)
                x_cat = x_cat.to(self.device)

                outputs = self.model(x_cont, x_cat)
                preds = torch.argmax(outputs, dim=1)

                # Move to CPU and convert to numpy
                all_preds.extend(preds.cpu().numpy())

        # Convert predictions back to original 1-7 range (model trained on 0-6)
        final_preds = np.array(all_preds) + 1

        # Create submission DataFrame
        submission = pd.DataFrame(
            {Config.ID_COL: self.test_ids, Config.TARGET_COL: final_preds}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save submission
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model(load_cached_data=True, debug_sample_size=None):
    """
    Main function to execute the training pipeline.
    """
    # 1. Set Seeds
    seed_everything(Config.SEED)

    # 2. Get DataLoaders
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    # 3. Determine Input Dimension
    # Fetch a single batch to inspect feature dimensions
    sample_cont, sample_cat, _ = next(iter(train_loader))
    num_cont_features = sample_cont.shape[1]
    print(f"Detected {num_cont_features} continuous features.")

    # 4. Initialize Model
    device = torch.device(Config.DEVICE)
    model = DCNV2(num_cont_features=num_cont_features)
    model.to(device)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_ids=test_ids,
        device=device,
    )

    # 6. Train
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # 7. Predict
    trainer.predict()
