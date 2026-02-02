import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, calculate_log_loss
from library.dataset import get_dataloaders
from library.model import get_model, setup_phase


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Dog Breed Classification task.
    Implements a two-phase training strategy (Head Adaptation -> Fine-Tuning).
    """

    def __init__(self):
        """
        Initializes the Trainer with model, device, and loss function.
        """
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)
        self.model = get_model(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        self.model = self.model.to(self.device)

        # CrossEntropyLoss expects raw logits (model output) and class indices
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, loader, optimizer):
        """
        Executes one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        return running_loss / count

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        Returns the average CrossEntropyLoss and the Multi Class Log Loss.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Apply Softmax to get probabilities for Log Loss calculation
                probs = torch.softmax(outputs, dim=1)

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Aggregate predictions
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate metrics
        avg_loss = running_loss / count
        log_loss_metric = calculate_log_loss(all_labels, all_preds)

        return avg_loss, log_loss_metric

    def fit(self, load_cached_data=True):
        """
        Orchestrates the two-phase training process.

        Args:
            load_cached_data (bool): Whether to load pre-processed metadata from cache.
        """
        # 1. Load Data
        print("Loading data...")
        train_loader, val_loader, test_loader, classes = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # 2. Phase 1: Head Adaptation
        print(
            f"\n--- Starting Phase 1: Head Adaptation ({Config.PHASE1_EPOCHS} Epochs) ---"
        )
        optimizer_p1 = setup_phase(self.model, "phase1")

        for epoch in range(Config.PHASE1_EPOCHS):
            train_loss = self.train_one_epoch(train_loader, optimizer_p1)
            val_loss, val_metric = self.validate(val_loader)
            print(
                f"Phase 1 Epoch {epoch+1}: Train Loss={train_loss}, Val Loss={val_loss}, Val LogLoss={val_metric}"
            )

        # 3. Phase 2: Fine-Tuning
        print(
            f"\n--- Starting Phase 2: Fine-Tuning ({Config.PHASE2_EPOCHS} Epochs) ---"
        )
        optimizer_p2 = setup_phase(self.model, "phase2")

        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(Config.PHASE2_EPOCHS):
            train_loss = self.train_one_epoch(train_loader, optimizer_p2)
            val_loss, val_metric = self.validate(val_loader)

            print(
                f"Phase 2 Epoch {epoch+1}: Train Loss={train_loss}, Val Loss={val_loss}, Val LogLoss={val_metric}"
            )

            # Early Stopping Check
            if val_metric < best_metric:
                best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved! (LogLoss: {best_metric})")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # 4. Generate Submission
        print("\nTraining complete. Generating submission with best model...")
        # Load best weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No best model file found. Using current model state.")

        self.generate_submission(test_loader, classes)

    def generate_submission(self, test_loader, classes):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Convert logits to probabilities
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                # Map probabilities to class names
                for img_id, prob_vector in zip(ids, probs):
                    row = {"id": img_id}
                    for idx, breed_name in enumerate(classes):
                        row[breed_name] = prob_vector[idx]
                    results.append(row)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Ensure column order: id, breed1, breed2, ... (alphabetical breeds)
        # classes list from get_dataloaders is already sorted
        cols = ["id"] + classes
        df = df[cols]

        # Save to CSV
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
