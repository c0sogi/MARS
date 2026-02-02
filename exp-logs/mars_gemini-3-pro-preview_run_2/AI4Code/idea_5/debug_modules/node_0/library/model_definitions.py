import os
import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoConfig

from library.config import (
    RIDGE_SOLVER,
    RIDGE_TOL,
    SEED,
    MODEL_NAME,
    DROPOUT,
    RIDGE_MODEL_PATH,
    TRANSFORMER_MODEL_PATH,
)
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(SEED)


class RidgeRegressorWrapper:
    """
    Wrapper for the Sparse Stream's Ridge Regression model.
    Encapsulates configuration, training, inference, and persistence.
    """

    def __init__(self, solver=RIDGE_SOLVER, tol=RIDGE_TOL, random_state=SEED):
        self.model = Ridge(solver=solver, tol=tol, random_state=random_state)
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fits the Ridge model to the sparse matrix X and targets y.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predicts regression scores for input X.
        """
        if not self.is_fitted:
            raise ValueError("Ridge model is not fitted yet.")
        return self.model.predict(X)

    def save(self, path=RIDGE_MODEL_PATH):
        """
        Saves the model to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Ridge model saved to {path}")

    def load(self, path=RIDGE_MODEL_PATH):
        """
        Loads the model from disk.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_fitted = True
            print(f"Ridge model loaded from {path}")
        else:
            print(
                f"Warning: No model found at {path}. Model initialized but not fitted."
            )
        return self


class TransformerRegressor(nn.Module):
    """
    PyTorch Module for the Dense Stream's Transformer model.
    Uses a pre-trained backbone (e.g., CodeBERT) and a linear regression head.
    """

    def __init__(self, model_name=MODEL_NAME, dropout_prob=DROPOUT):
        super(TransformerRegressor, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # Linear regression head: Hidden Size -> 1 scalar score
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        Returns a 1D tensor of scores with shape [batch_size].
        """
        # Pass inputs through the transformer backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token representation (first token of the last hidden state)
        # Shape: [batch_size, hidden_size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        x = self.dropout(cls_embedding)

        # Project to scalar
        # Shape: [batch_size, 1]
        logits = self.regressor(x)

        # Squeeze to return [batch_size] for compatibility with MSELoss
        return logits.squeeze(-1)

    def save(self, path=TRANSFORMER_MODEL_PATH):
        """
        Saves the model state dictionary to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Transformer model saved to {path}")

    def load(self, path=TRANSFORMER_MODEL_PATH, device="cpu"):
        """
        Loads the model state dictionary from disk.
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            print(f"Transformer model loaded from {path}")
        else:
            print(
                f"Warning: No model found at {path}. Model initialized with pre-trained weights."
            )
        return self
