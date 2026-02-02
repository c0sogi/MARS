import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import SimpleMLP


class ModelTrainer:
    """
    Handles the training, validation, and inference processes for the Bird Species Prediction model.
    """

    def __init__(self, device=None):
        """
        Initializes the trainer with the model and device.
        """
        Config.set_seed(Config.SEED)
        self.device = device if device else Config.get_device()
        self.model = SimpleMLP().to(self.device)
        self.best_model_state = None

    def _robust_roc_auc(self, y_true, y_score):
        """
        Calculates macro-averaged ROC AUC robustly, ignoring classes with only one label in the batch.
        """
        aucs = []
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            # Check if both classes (0 and 1) are present
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[:, i], y_score[:, i])
                    aucs.append(auc)
                except ValueError:
                    pass

        if not aucs:
            return 0.5  # Default fallback if no classes can be evaluated
        return np.mean(aucs)

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
    ):
        """
        Trains the model with Early Stopping based on Validation AUC.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            num_epochs (int): Maximum number of epochs.
            lr (float): Learning rate.
            patience (int): Early stopping patience.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = -1.0
        patience_counter = 0

        print("Starting training...")

        for epoch in range(num_epochs):
            # --- Training Phase ---
            self.model.train()
            train_losses = []

            for features, labels, _ in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # --- Validation Phase ---
            self.model.eval()
            val_losses = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for features, labels, _ in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(features)
                    loss = criterion(logits, labels)
                    val_losses.append(loss.item())

                    # Apply sigmoid for probabilities
                    probs = torch.sigmoid(logits)

                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())

            avg_val_loss = np.mean(val_losses)
            all_labels = np.vstack(all_labels)
            all_probs = np.vstack(all_probs)

            val_auc = self._robust_roc_auc(all_labels, all_probs)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Val AUC: {val_auc}"
            )

            # --- Early Stopping Check ---
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model for inference
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with Val AUC: {best_val_auc}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            tuple: (predictions, rec_ids)
                predictions (np.ndarray): Probability matrix (N, num_classes).
                rec_ids (np.ndarray): Array of recording IDs (N,).
        """
        self.model.eval()
        all_probs = []
        all_ids = []

        with torch.no_grad():
            for features, _, rec_ids in test_loader:
                features = features.to(self.device)

                logits = self.model(features)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_ids.append(rec_ids.cpu().numpy())

        if not all_probs:
            return np.array([]), np.array([])

        return np.vstack(all_probs), np.concatenate(all_ids)

    def generate_submission(
        self, predictions, test_ids, output_path=Config.SUBMISSION_FILE_PATH
    ):
        """
        Formats predictions and saves the submission CSV.

        Args:
            predictions (np.ndarray): Predicted probabilities (N_samples, N_classes).
            test_ids (np.ndarray): Recording IDs corresponding to the predictions.
            output_path (str): Path to save the CSV.
        """
        submission_rows = []
        num_classes = predictions.shape[1]

        # Ensure input arrays are aligned
        if len(predictions) != len(test_ids):
            raise ValueError(
                f"Mismatch between predictions ({len(predictions)}) and IDs ({len(test_ids)})"
            )

        for i, rec_id in enumerate(test_ids):
            probs = predictions[i]
            for species_idx in range(num_classes):
                # Construct Id: rec_id * 100 + species_idx
                row_id = int(rec_id * Config.ID_MULTIPLIER + species_idx)
                prob = probs[species_idx]
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
