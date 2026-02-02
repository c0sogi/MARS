import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    POS_WEIGHT,
    WORKING_DIR,
)
from library.feature_extractor import ISICNet


class MalignancyClassifier:
    """
    Wrapper for training and inference of the ISICNet PyTorch model.
    """

    def __init__(self, num_tabular_features):
        self.model = ISICNet(num_tabular_features).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS
        )

    def fit(self, train_loader, val_loader):
        print(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")

        best_auc = 0.0

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            train_loss_accum = 0.0
            all_preds = []
            all_targets = []

            for images, tabular, targets in train_loader:
                images = images.to(DEVICE)
                tabular = tabular.to(DEVICE)
                targets = targets.to(DEVICE).unsqueeze(1)  # (B, 1)

                self.optimizer.zero_grad()
                logits = self.model(images, tabular)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                train_loss_accum += loss.item() * images.size(0)

                # Store for metrics
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.append(probs)
                all_targets.append(targets.cpu().numpy())

            self.scheduler.step()

            # Epoch Metrics
            train_loss = train_loss_accum / len(train_loader.dataset)
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            try:
                train_auc = roc_auc_score(all_targets, all_preds)
            except:
                train_auc = 0.5

            # Validation
            val_auc, val_loss, _ = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                self.save("best_model.pth")

        print(f"Training complete. Best Val AUC: {best_auc:.4f}")
        # Load best model
        self.load("best_model.pth")

    def evaluate(self, dataloader):
        self.model.eval()
        loss_accum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, tabular, targets in dataloader:
                images = images.to(DEVICE)
                tabular = tabular.to(DEVICE)
                targets = targets.to(DEVICE).unsqueeze(1)

                logits = self.model(images, tabular)
                loss = self.criterion(logits, targets)

                loss_accum += loss.item() * images.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_preds.append(probs)
                all_targets.append(targets.cpu().numpy())

        avg_loss = loss_accum / len(dataloader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5

        return auc, avg_loss, all_preds.flatten()

    def predict_proba(self, dataloader):
        _, _, preds = self.evaluate(dataloader)
        return preds

    def save(self, filename):
        path = os.path.join(WORKING_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def load(self, filename):
        path = os.path.join(WORKING_DIR, filename)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        return self
