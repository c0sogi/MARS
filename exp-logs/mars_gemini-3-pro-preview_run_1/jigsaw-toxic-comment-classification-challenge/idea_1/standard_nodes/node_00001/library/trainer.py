import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, load_embeddings
from library.data_loader import get_dataloaders
from library.model import BiGRU_Pool_Net


class Trainer:
    """
    Trainer class to manage training, validation, and prediction for the Toxicity Classification task.
    """

    def __init__(self, debug=False, load_cached_data=True):
        self.debug = debug
        self.load_cached_data = load_cached_data
        self.device = Config.DEVICE

        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()

        # Data components
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.word_index = None

        # Training state
        self.best_val_auc = 0.0

    def load_data(self):
        """Loads dataloaders and word index."""
        print("Loading Data...")
        self.train_loader, self.val_loader, self.test_loader, self.word_index = (
            get_dataloaders(
                debug=self.debug,
                batch_size=Config.BATCH_SIZE,
                load_cached_data=self.load_cached_data,
            )
        )

    def build_model(self):
        """Initializes the model, embeddings, and optimizer."""
        print("Building Model...")

        # Ensure data is loaded to get word_index
        if self.word_index is None:
            self.load_data()

        # Load pre-trained embeddings if available
        embedding_path = os.path.join(Config.INPUT_DIR, "glove.840B.300d.txt")
        # Check if file exists, otherwise pass None to initialize randomly
        if not os.path.exists(embedding_path):
            embedding_path = None

        embedding_matrix = load_embeddings(
            embedding_path, self.word_index, Config.EMBED_DIM
        )

        # Initialize Model
        self.model = BiGRU_Pool_Net(
            vocab_size=Config.MAX_FEATURES,
            embed_dim=Config.EMBED_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            output_dim=Config.NUM_CLASSES,
            embedding_matrix=embedding_matrix,
            dropout=Config.DROPOUT,
        )
        self.model.to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Store for AUC
            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Calculate Mean Column-wise ROC AUC
        auc_scores = []
        for i in range(all_targets.shape[1]):
            try:
                if len(np.unique(all_targets[:, i])) > 1:
                    score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                else:
                    score = 0.5
                auc_scores.append(score)
            except ValueError:
                auc_scores.append(0.5)

        epoch_auc = np.mean(auc_scores)
        return epoch_loss, epoch_auc

    def validate(self):
        """Runs validation."""
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * inputs.size(0)
                all_targets.append(targets.cpu().numpy())
                all_preds.append(torch.sigmoid(logits).cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        auc_scores = []
        for i in range(all_targets.shape[1]):
            try:
                if len(np.unique(all_targets[:, i])) > 1:
                    score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                else:
                    score = 0.5
                auc_scores.append(score)
            except ValueError:
                auc_scores.append(0.5)

        epoch_auc = np.mean(auc_scores)
        return epoch_loss, epoch_auc

    def fit(self):
        """Main training loop with early stopping."""
        seed_everything(Config.SEED)

        if self.model is None:
            self.build_model()

        print("Starting Training...")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc = self.validate()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - "
                f"Time: {elapsed:.0f}s - "
                f"Train Loss: {train_loss} - Train AUC: {train_auc} - "
                f"Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Checkpointing & Early Stopping
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Best Validation AUC: {self.best_val_auc}")

    def predict(self):
        """Generates predictions on the test set using the best model."""
        print("Generating Predictions...")

        # Load best model weights
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print(
                f"Model file {Config.MODEL_SAVE_PATH} not found. Ensure training has run."
            )
            return

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds = []
        with torch.no_grad():
            for inputs in self.test_loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                preds = torch.sigmoid(logits)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds)

    def save_submission(self):
        """Generates predictions and saves the submission file."""
        if self.test_loader is None:
            self.load_data()

        test_preds = self.predict()

        if test_preds is not None:
            submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
            submission[Config.LABEL_COLS] = test_preds
            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
