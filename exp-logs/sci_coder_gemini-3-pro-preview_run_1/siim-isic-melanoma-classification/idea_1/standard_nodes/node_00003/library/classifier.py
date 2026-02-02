import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    POS_WEIGHT,
    WORKING_DIR,
)
from library.feature_extractor import SkinLesionModel


class MalignancyClassifier:
    """
    A wrapper around the PyTorch SkinLesionModel to handle training and inference.
    """

    def __init__(self, tabular_dim):
        self.model = SkinLesionModel(tabular_dim).to(DEVICE)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        # Use weighted BCE loss to handle class imbalance
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
        )

    def fit(self, train_loader, val_loader=None):
        """
        Trains the PyTorch model.
        """
        print(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            train_loss = 0

            for images, tabular, targets in train_loader:
                images = images.to(DEVICE)
                tabular = tabular.to(DEVICE)
                targets = targets.to(DEVICE).unsqueeze(1)

                self.optimizer.zero_grad()
                logits = self.model(images, tabular)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_train_loss:.4f}")

            if val_loader:
                val_auc = self.evaluate(val_loader)
                print(f"  Val AUC: {val_auc:.4f}")

    def evaluate(self, loader):
        """
        Evaluates the model and returns ROC AUC.
        """
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, tabular, targets in loader:
                images = images.to(DEVICE)
                tabular = tabular.to(DEVICE)

                logits = self.model(images, tabular)
                probs = torch.sigmoid(logits)

                all_probs.extend(probs.cpu().numpy().flatten())
                all_targets.extend(targets.numpy().flatten())

        return roc_auc_score(all_targets, all_probs)

    def predict_proba(self, loader):
        """
        Predicts probabilities for the positive class.
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for images, tabular, _ in loader:
                images = images.to(DEVICE)
                tabular = tabular.to(DEVICE)

                logits = self.model(images, tabular)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())

        return np.array(all_probs)

    def save(self, filename="model.pth"):
        path = os.path.join(WORKING_DIR, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, filename="model.pth"):
        path = os.path.join(WORKING_DIR, filename)
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        return self
