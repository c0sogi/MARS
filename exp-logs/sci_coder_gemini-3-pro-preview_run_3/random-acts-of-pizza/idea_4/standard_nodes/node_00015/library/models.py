import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config


class PizzaTransformer(nn.Module):
    """
    PyTorch Module wrapping a Transformer backbone with a classification head.
    """

    def __init__(self, model_name=Config.BERT_MODEL_NAME, dropout=0.1):
        super(PizzaTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Pass through transformer
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token (first token of last hidden state)
        # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classification head
        x = self.dropout(cls_token)
        logits = self.classifier(x)

        return logits


class SemanticFineTuner:
    """
    Wrapper class to handle training and inference for the PizzaTransformer.
    Mimics an sklearn-like interface (fit, predict_proba).
    """

    def __init__(self):
        self.device = torch.device(Config.BERT_TRAIN_PARAMS["device"])
        self.model = PizzaTransformer().to(self.device)
        self.batch_size = Config.BERT_TRAIN_PARAMS["batch_size"]
        self.epochs = Config.BERT_TRAIN_PARAMS["epochs"]
        self.learning_rate = Config.BERT_TRAIN_PARAMS["learning_rate"]
        self.patience = Config.BERT_TRAIN_PARAMS["early_stopping_patience"]
        self.weight_decay = Config.BERT_TRAIN_PARAMS["weight_decay"]

    def fit(self, train_loader, val_loader):
        """
        Trains the model with Early Stopping.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        print(f"Starting training on device: {self.device}")

        for epoch in range(self.epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * input_ids.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device).unsqueeze(1)

                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * input_ids.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.8f} - "
                f"Val Loss: {avg_val_loss:.8f}"
            )

            # --- Early Stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def predict_proba(self, loader):
        """
        Generates probability predictions (class 1).
        """
        self.model.eval()
        probs_list = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())

        return np.vstack(probs_list).flatten()


def get_lexical_model():
    """
    Returns the Random Forest model for the Lexical View (TF-IDF).
    """
    return RandomForestClassifier(**Config.RF_PARAMS)


def get_style_model():
    """
    Returns the XGBoost model for the Style/Contextual View.
    """
    return XGBClassifier(**Config.XGB_PARAMS)


def get_meta_model():
    """
    Returns the Logistic Regression model for the Stacking Meta-Learner.
    """
    return LogisticRegression(**Config.META_PARAMS)
