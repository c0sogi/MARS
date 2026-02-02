import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from library.config import Config
from library.dataset import get_dataloaders
from library.model import get_model


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self):
        # Set seed for reproducibility
        set_seed()

        self.device = Config.DEVICE
        self.model = get_model(
            pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=True
        )

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Accumulate loss weighted by batch size
            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        count = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / count if count > 0 else 0.0
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        return val_loss, val_f1

    def fit(self):
        print(f"Starting training on device: {self.device}")
        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch()
            val_loss, val_f1 = self.validate()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_f1}"
            )

            # Checkpoint logic
            if val_f1 > best_f1 + Config.EARLY_STOPPING_MIN_DELTA:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved with F1: {best_f1}")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training completed. Best Val F1: {best_f1}")

    def generate_submission(self):
        # Load the best model weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model for submission generation.")
        else:
            print(
                "Warning: No checkpoint found. Generating submission with current model weights."
            )

        self.model.eval()
        results = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for images, ids in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                for id_val, pred_val in zip(ids, preds):
                    results.append({"Id": id_val, "Predicted": pred_val})

        # Create DataFrame and save
        df = pd.DataFrame(results)

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
