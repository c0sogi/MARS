import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss
from library.config import (
    DEVICE,
    SEED,
    WORKING_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
)
from library.utils import set_seed
from library.feature_extractor import DogResNet


class DeepClassifier:
    """
    Trainer for the PyTorch DogResNet model.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        set_seed(SEED)
        self.device = DEVICE
        self.model = DogResNet(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    def train(self, train_loader, val_loader):
        print(f"Starting fine-tuning for {NUM_EPOCHS} epochs...")
        best_val_loss = float("inf")

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            val_loss = self.evaluate_loss(val_loader)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)

        print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
        # Load best model
        self.model.load_state_dict(torch.load(self.best_model_path))

    def evaluate_loss(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
        return total_loss / len(val_loader.dataset)

    def evaluate(self, val_loader):
        """
        Evaluates the model on validation data using sklearn Log Loss.
        Returns loss and predictions/labels for failure analysis.
        """
        print("Evaluating model on validation set...")
        self.model.eval()
        probs_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                probs_list.append(probs.cpu().numpy())
                labels_list.append(labels.numpy())

        y_pred_proba = np.vstack(probs_list)
        y_val = np.concatenate(labels_list)

        # Calculate Log Loss
        loss = log_loss(y_val, y_pred_proba, labels=list(range(NUM_CLASSES)))
        print(f"Validation Log Loss: {loss}")
        return loss, y_pred_proba, y_val

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        print("Generating predictions...")
        self.model.eval()
        probs_list = []
        ids_list = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                probs_list.append(probs.cpu().numpy())
                ids_list.extend(ids)

        return np.vstack(probs_list), ids_list
