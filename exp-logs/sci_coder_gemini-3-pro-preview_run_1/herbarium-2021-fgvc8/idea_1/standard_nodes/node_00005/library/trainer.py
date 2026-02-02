import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import pandas as pd

from library.config import Config
from library.dataset import get_dataloaders
from library.model import get_model


class Trainer:
    def __init__(self, debug=False):
        """
        Initializes the Trainer with model, dataloaders, optimizer, and scheduler.
        """
        self.debug = debug
        self.set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # Load DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=self.debug
        )

        # Initialize Model
        self.model = get_model(pretrained=Config.PRETRAINED)
        self.model.to(self.device)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.epochs = Config.NUM_EPOCHS
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Early Stopping and Checkpointing
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.model_path = os.path.join(Config.IDEA_DIR, "model.pth")

    def set_seed(self, seed):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_one_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        running_loss = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def evaluate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            # Calculate Macro F1 Score
            f1 = f1_score(all_labels, all_preds, average="macro")
        else:
            f1 = 0.0

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss, f1

    def train(self):
        """Runs the full training loop with early stopping."""
        print(f"Starting training for {self.epochs} epochs on {self.device}...")
        start_time = time.time()

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_f1 = self.evaluate()

            print(
                f"Epoch {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
            )

            # Early Stopping Check (Monitor Macro F1)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Saved Best Model (F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.PATIENCE:
                print("Early Stopping Triggered")
                break

        print(f"Training finished in {time.time() - start_time} seconds")

    def infer_and_save(self):
        """
        Loads the best model, generates predictions for the test set,
        and saves the submission file.
        """
        print("Starting inference...")

        # Load best model weights
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            print("Loaded best model from checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        # Load class mapping (created during dataset initialization)
        classes_path = os.path.join(Config.IDEA_DIR, "classes.npy")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(
                "classes.npy not found. Ensure training/dataloaders ran first."
            )

        unique_cats = np.load(classes_path)

        predictions = []
        image_ids = []

        with torch.no_grad():
            for images, ids in self.test_loader:
                images = images.to(self.device)

                with autocast():
                    outputs = self.model(images)

                # Get predicted indices
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                # Map indices back to original category_ids
                mapped_preds = unique_cats[preds]

                predictions.extend(mapped_preds)
                image_ids.extend(ids.numpy())

        # Create submission dataframe
        sub_df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

        # Save submission
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
