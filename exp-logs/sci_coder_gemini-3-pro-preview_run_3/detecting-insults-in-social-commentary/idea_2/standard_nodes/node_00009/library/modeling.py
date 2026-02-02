import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import RobertaModel, get_linear_schedule_with_warmup
import copy
from library.utils import set_seed


class NBSVM(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes - Support Vector Machine (or Logistic Regression) hybrid.
    Scales features by the Naive Bayes log-count ratio before feeding them
    into a linear classifier.
    """

    def __init__(self, C=1.0, dual=False, n_jobs=1, random_state=42):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.r = None
        self.clf = None

    def fit(self, X, y):
        """
        Fit the NBSVM model.

        Args:
            X: Dense feature matrix (TF-IDF).
            y: Target labels.
        """
        set_seed(self.random_state)

        # Ensure y is integer for indexing
        y = y.astype(int)

        # Compute Naive Bayes log-count ratios
        # p: Sum of features for the positive class (Insult)
        # q: Sum of features for the negative class (Neutral)
        # We add 1.0 for Laplace smoothing
        p = np.sum(X[y == 1], axis=0) + 1.0
        q = np.sum(X[y == 0], axis=0) + 1.0

        # Calculate log-count ratio r
        # r = log( (p / ||p||) / (q / ||q||) )
        self.r = np.log((p / np.sum(p)) / (q / np.sum(q)))

        # Scale the feature matrix
        X_nb = X * self.r

        # Train Logistic Regression on scaled features
        # We use 'liblinear' if dual=True (good for high dim), else 'lbfgs'
        solver = "liblinear" if self.dual else "lbfgs"
        self.clf = LogisticRegression(
            C=self.C,
            dual=self.dual,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            solver=solver,
            max_iter=1000,
        )
        self.clf.fit(X_nb, y)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if self.clf is None or self.r is None:
            raise RuntimeError("Model not fitted")

        # Scale features using the learned ratio
        X_nb = X * self.r
        return self.clf.predict_proba(X_nb)

    def predict(self, X):
        """
        Predict class labels.
        """
        if self.clf is None or self.r is None:
            raise RuntimeError("Model not fitted")

        X_nb = X * self.r
        return self.clf.predict(X_nb)


class RoBERTaClassifier(nn.Module):
    """
    Fine-Tuned RoBERTa Model for Binary Classification.
    """

    def __init__(self, model_name="roberta-large", dropout=0.15, num_labels=1):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass with Mean Pooling.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Mean Pooling (Cite solution_lesson_node_00006: Improving Transformer architecture)
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        output = self.dropout(mean_embeddings)
        logits = self.classifier(output)
        return logits

    def train_model(
        self, train_loader, val_loader, device, epochs=3, lr=2e-5, patience=2
    ):
        """
        Runs the training loop with Early Stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: torch.device.
            epochs: Maximum number of epochs.
            lr: Learning rate.
            patience: Early stopping patience.

        Returns:
            best_val_auc: The best AUC score achieved on the validation set.
        """
        set_seed(42)
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0.0
        best_model_wts = copy.deepcopy(self.state_dict())
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # --- Training Phase ---
            self.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self(input_ids, attention_mask)
                loss = criterion(logits, targets)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * input_ids.size(0)

            train_loss /= len(train_loader.dataset)

            # --- Validation Phase ---
            self.eval()
            val_preds = []
            val_targets = []
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["target"].to(device).unsqueeze(1)

                    logits = self(input_ids, attention_mask)
                    loss = criterion(logits, targets)
                    val_loss += loss.item() * input_ids.size(0)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(targets.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_auc = roc_auc_score(val_targets, val_preds)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # --- Early Stopping Logic ---
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_wts = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model weights
        self.load_state_dict(best_model_wts)
        print(f"Training complete. Best Val AUC: {best_val_auc}")
        return best_val_auc

    def predict(self, data_loader, device):
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for inference.
            device: torch.device.

        Returns:
            np.array: Flattened array of probability scores.
        """
        self.eval()
        self.to(device)
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                logits = self(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs)

        return np.array(predictions).flatten()
