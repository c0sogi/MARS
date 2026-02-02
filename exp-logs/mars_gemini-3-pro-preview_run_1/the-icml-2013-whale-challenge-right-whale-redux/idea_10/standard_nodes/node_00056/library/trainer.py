import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.dataset import get_dataloaders
from library.csk_resnet import CoordAttResNet18CRNN
from library.utils import set_seed, compute_auc


class Trainer:
    """
    Manages the training, validation, and prediction processes for the Whale Detection model.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = CoordAttResNet18CRNN().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        # Criterion with Class Weighting
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def mixup_data(self, x, y, alpha=0.4):
        """Applies Mixup augmentation to the batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Computes the Mixup loss."""
        y_a = y_a.view(-1, 1)
        y_b = y_b.view(-1, 1)
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_one_epoch(self, loader):
        """Trains the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        num_samples = 0

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)

            # Apply Mixup
            data, target_a, target_b, lam = self.mixup_data(
                data, target, alpha=Config.MIXUP_ALPHA
            )

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.mixup_criterion(output, target_a, target_b, lam)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            num_samples += batch_size

        return running_loss / num_samples if num_samples > 0 else 0.0

    def validate(self, loader):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        num_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)

                output = self.model(data)
                target_view = target.view(-1, 1)

                loss = self.criterion(output, target_view)
                running_loss += loss.item() * batch_size
                num_samples += batch_size

                preds = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(target.cpu().numpy())

        avg_loss = running_loss / num_samples if num_samples > 0 else 0.0
        auc = compute_auc(all_targets, all_preds)
        return avg_loss, auc

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """Runs the training loop with Early Stopping."""
        best_auc = 0.0
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        patience = 5
        patience_counter = 0

        print("Starting Training...")
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            self.scheduler.step(val_auc)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training Complete. Best Validation AUC: {best_auc}")
        # Load best model for inference
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )

    def predict(self, test_loader):
        """Generates predictions for the test set."""
        self.model.eval()
        predictions = []
        clips = []

        with torch.no_grad():
            for data, clip_ids in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                preds = torch.sigmoid(output).cpu().numpy()

                predictions.extend(preds.flatten())
                clips.extend(clip_ids)

        return clips, predictions


def run_training(epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Main execution function to setup environment, train model, and generate submission.
    """
    # Configure debug mode
    Config.DEBUG = debug

    set_seed(Config.SEED)

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Running on device: {Config.DEVICE}")
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    trainer = Trainer()
    trainer.fit(train_loader, val_loader, epochs=epochs)

    print("Generating predictions on test set...")
    clips, probs = trainer.predict(test_loader)

    submission_df = pd.DataFrame({"clip": clips, "probability": probs})
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
